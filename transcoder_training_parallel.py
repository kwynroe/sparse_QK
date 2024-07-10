from functools import partial

import numpy as np
import plotly.express as px
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import einops
import wandb
from ActivationStoreParallel import ActivationsStore
from optimize import get_scheduler
from sparse_transcoder import SparseTranscoder
from metrics_training import WandbLogger

def apply_causal_mask(attn_scores):
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)).cuda(), diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, 1e-9)
        return attn_scores




def train_transcoder_on_language_model_parallel(
    cfg,
    model: HookedTransformer,
    query_transcoder: SparseTranscoder,
    key_transcoder: SparseTranscoder,
    activation_store: ActivationsStore,
):
    # if feature_sampling_method is not None:    # TODO: seems not to be used?
    #     feature_sampling_method = feature_sampling_method.lower()

    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.train_batch_size
    n_training_steps = 0
    n_training_tokens = 0
    if cfg.n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // cfg.n_checkpoints))[1:]
    
    # track active features TODO: think these are no longer used
    # n_forward_passes_since_fired_q = torch.zeros(query_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
    # n_forward_passes_since_fired_k = torch.zeros(key_transcoder.cfg.d_hidden, device=key_transcoder.cfg.device)
    
    # Optimizer and scheduler.
    optimizer = Adam(list(query_transcoder.parameters()) + list(key_transcoder.parameters()),
                     lr = query_transcoder.cfg.lr)
    scheduler = get_scheduler(
        query_transcoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = query_transcoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=query_transcoder.cfg.lr / 10, # heuristic for now. 
    )

    print("Initializing autoencoders.")
    W_Q, b_Q = model.W_Q[cfg.layer], model.b_Q[cfg.layer]
    W_K, b_K = model.W_K[cfg.layer], model.b_K[cfg.layer]
    query_transcoder.initialize_b_dec(activation_store, W_Q, b_Q)
    key_transcoder.initialize_b_dec(activation_store, W_K, b_K)

    query_transcoder.train()
    key_transcoder.train()
    
    if query_transcoder.cfg.attn_scores_normed:
        attn_scores_norm = query_transcoder.d_head**0.5
    else:
        attn_scores_norm = 1
        
    # Initialise logging.
    wandb_logger = WandbLogger(cfg, query_transcoder, key_transcoder, optimizer)

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        
        # Do a training step. - TODO: think these might be redundant?
        query_transcoder.train()
        key_transcoder.train()

        # Make sure the W_dec is still zero-norm
        query_transcoder.set_decoder_norm_to_unit_norm()
        key_transcoder.set_decoder_norm_to_unit_norm()
           
        scheduler.step()
        optimizer.zero_grad()
        
        data = activation_store.next_batch()
        data = einops.rearrange(data, "(batch posn) d_model -> batch posn d_model", posn = cfg.context_size)
        
        #ground truth activations for loss
        true_queries = einops.einsum(model.W_Q[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[query_transcoder.layer]
        true_keys = einops.einsum(model.W_K[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[query_transcoder.layer]
        true_scores = einops.einsum(true_queries, true_keys, " ... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")/attn_scores_norm
        true_patt = apply_causal_mask(true_scores).log_softmax(-1)
        
        true_queries_flatt = einops.rearrange(true_queries, " ... n_head d_head -> ... (n_head d_head)")
        true_keys_flatt = einops.rearrange(true_keys, " ... n_head d_head -> ... (n_head d_head)")
        
        
        # Forward and Backward Passes
        reconstr_queries, feature_actsQ, mse_lossQ, reg_lossQ= query_transcoder(
            data,
            true_queries_flatt
        )
        
        reconstr_keys, feature_actsK,  mse_lossK, reg_lossK = key_transcoder(
            data,
            true_keys_flatt
        )
        
        reconstr_queries = einops.rearrange(reconstr_queries, " ... (n_head d_head) -> ... n_head d_head", n_head = 12)
        reconstr_keys = einops.rearrange(reconstr_keys, " ... (n_head d_head) -> ... n_head d_head", n_head = 12)
        
        #Calculate attention scores using reconstructed components (full reconstructed + half reconstructed)
        pred_attn_scores_true_keys = einops.einsum(reconstr_queries, true_keys, " ... posnq n_head d_head, ... posnk n_head d_head -> ... n_head posnq posnk")/attn_scores_norm
        pred_attn_scores_true_queries = einops.einsum(reconstr_keys, true_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")/attn_scores_norm
        full_pred_attn_scores = einops.einsum(reconstr_keys, reconstr_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")/attn_scores_norm
        
        #Error on attention scores
        attn_score_loss_true_keys = ((pred_attn_scores_true_keys) - (true_scores)).pow(2).mean()
        attn_score_loss_true_queries = ((pred_attn_scores_true_queries) - (true_scores)).pow(2).mean()
        attn_scores_loss_full_pred = ((full_pred_attn_scores) - (true_scores)).pow(2).mean()
        patt_true_keys = apply_causal_mask(pred_attn_scores_true_keys).log_softmax(-1)
        patt_true_queries = apply_causal_mask(pred_attn_scores_true_queries).log_softmax(-1)
        patt_full_reconstr = apply_causal_mask(full_pred_attn_scores).log_softmax(-1)                              

        #Reshape patterns and compute KL-Divergences for loss
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target = True)
        true_patt_flat = true_patt.view((-1, true_patt.shape[-1]))
        patt_full_reconstr = patt_full_reconstr.view((-1, patt_full_reconstr.shape[-1]))
        patt_true_keys = patt_true_keys.view((-1, patt_true_keys.shape[-1]))
        patt_true_queries = patt_true_queries.view((-1, patt_true_queries.shape[-1]))
        
        patt_loss_true_queries = kl_loss(patt_true_queries, true_patt_flat)
        patt_loss_true_keys = kl_loss(patt_true_keys, true_patt_flat)
        patt_loss_full_pred = kl_loss(patt_full_reconstr, true_patt_flat)
        
        
        #Full loss
        loss = 3*patt_loss_full_pred + patt_loss_true_queries + patt_loss_true_keys + reg_lossQ + reg_lossK
        
        #Feature Sparsity Calculations - TODO: not used?
        # did_fireQ = ((feature_actsQ > 0).float().sum(0).sum(0) > 0)
        # did_fireK = ((feature_actsK > 0).float().sum(0).sum(0) > 0)
        # n_forward_passes_since_fired_q += 1
        # n_forward_passes_since_fired_q[did_fireQ] = 0
        # n_forward_passes_since_fired_k += 1
        # n_forward_passes_since_fired_k[did_fireK] = 0
        n_training_tokens += cfg.train_batch_size

        if cfg.log_to_wandb and (n_training_steps + 1) % cfg.wandb_log_frequency == 0:
            print("Logging to wandb")
            wandb_logger.log_to_wandb(
                feature_actsQ,
                feature_actsK,
                mse_lossQ,
                mse_lossK,
                reg_lossQ,
                reg_lossK,
                true_scores,
                n_training_tokens,
                reconstr_keys,
                reconstr_queries,
                patt_loss_true_keys,
                patt_loss_true_queries,
                patt_loss_full_pred,
                attn_scores_loss_full_pred,
                attn_score_loss_true_keys,
                attn_score_loss_true_queries,
                true_patt_flat,
                patt_full_reconstr,
                pred_attn_scores_true_keys,
                pred_attn_scores_true_queries,
                n_training_steps,
            )

        pbar.set_description(
            f"{n_training_steps}| MSE Loss {loss.item():.3f}"
        )
        pbar.update(cfg.train_batch_size)

        loss.backward()
        #query_transcoder.remove_gradient_parallel_to_decoder_directions()
        #key_transcoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


        # checkpoint if at checkpoint frequency
        if cfg.n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = query_transcoder.cfg
            path_q = f"{query_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{query_transcoder.get_name()}.pt"
            path_k = f"{key_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{key_transcoder.get_name()}.pt"
            #log_feature_sparsity_path = f"{sparse_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder.get_name()}_log_feature_sparsity.pt"
            query_transcoder.save_model(path_q)
            key_transcoder.save_model(path_k)
            #torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                cfg.n_checkpoints = 0        
        n_training_steps += 1

    return query_transcoder, key_transcoder

