from functools import partial

import numpy as np
import plotly.express as px
import torch
from torch import Tensor, nn
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import einops
import wandb
from ActivationStoreParallel import ActivationsStore
from optimize import get_scheduler
from sparse_transcoder import SparseTranscoder
import torch.nn.functional as F


def train_transcoder_on_language_model_parallel(
    cfg,
    model,
    query_transcoder,
    key_transcoder,
    activation_store
):
    print("TRAIN STARTED")
    batch_size = cfg.batch_size
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0
    steps_before_reset = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features

    mask = nn.Parameter(torch.ones(query_transcoder.d_hidden, key_transcoder.d_hidden, query_transcoder.n_heads))
    optimizer = Adam(mask.parameters(), lr = cfg.lr)
    
    print("gonna schedule!")
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=cfg.lr / 10, # heuristic for now. 
    )
    
    if cfg.attn_scores_norm:
        attn_scores_norm = query_transcoder.d_head**0.5
    else:
        attn_scores_norm = 1
        
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        mask = torch.clip(mask, min = 0, max = 1)
        scheduler.step()
        optimizer.zero_grad()
        
        data = activation_store.next_batch()
        data = einops.rearrange(data, "(batch posn) d_model -> batch posn d_model", posn = cfg.context_size)
        true_queries = einops.einsum(model.W_Q[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[query_transcoder.layer]
        true_keys = einops.einsum(model.W_K[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[query_transcoder.layer]
        true_queries_flat = einops.rearrange(true_queries, " ... n_head d_head -> ... (n_head d_head)")
        true_keys_flat = einops.rearrange(true_keys, " ... n_head d_head -> ... (n_head d_head)")
        true_scores = einops.einsum(true_queries, true_keys, " ... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")/attn_scores_norm
        true_patt = true_scores.log_softmax(-1)
        
        q_features = einops.rearrange(query_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = query_transcoder.d_head)
        k_features = einops.rearrange(key_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = key_transcoder.d_head)
        feature_map = einops.einsum(q_features, k_features, "d_hidden_Q n_head d_head, d_hidden_K n_head d_head -> d_hidden_Q d_hidden_K n_head")
        bias_acts = einops.einsum(k_features, query_transcoder.b_dec, "d_hidden_K n_head d_head, n_head d_head -> n_head d_hidden_K")

        feature_acts_Q = F.relu(einops.einsum((true_queries_flat - query_transcoder.b_dec), query_transcoder.W_enc, "... d_model, d_model d_hidden -> ... d_hidden") + query_transcoder.b_enc)
        feature_acts_K = F.relu(einops.einsum((true_keys_flat - key_transcoder.b_dec), key_transcoder.W_enc, "... d_model, d_model d_hidden -> ... d_hidden") + key_transcoder.b_enc)

        #given feature acts and map between features, compute attention contribution
        
        attn_contribution = einops.einsum(feature_acts_Q, feature_map, "... d_hidden Q, d_hidden_Q d_hidden_K n_head -> ... d_hidden_K n_head")
        attn_contribution = einops.einsum(attn_contribution, feature_acts_K, "... d_hidden_K n_head, ... d_hidden_K -> ... n_head")
        contr_from_bias = einops.einsum(bias_acts, feature_acts_K, "n_head d_hidden_K, ... d_hidden_K -> ... n_head")
        attn_scores_reconstr = attn_contribution + contr_from_bias
        
        reconstr_patt = attn_scores_reconstr.log_softmax(-1)
        true_patt_flat = true_patt.view((-1, true_patt.shape[-1]))
        reconstr_patt = reconstr_patt.view((-1, reconstr_patt.shape[-1]))
        kl = torch.nn.KLDivLoss(reduction = "batchmean", log_target = True)
        kl_div = kl(reconstr_patt, true_patt_flat)
        reg_loss = cfg.reg_coefficient * torch.sqrt(mask.float().abs() + cfg.eps).sum()
        loss = kl_div + reg_loss
        
        n_training_tokens += batch_size

        with torch.no_grad():
            frac_zeros = (mask == 0).mean()
            wandb.log(
                    {
                        # losses
                        "losses/kl_div": kl_div.item(),
                        # variance explained
                        "metrics/frac_zeros" : frac_zeros.item(),

                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            """if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                sparse_transcoder.eval()
                run_evals(sparse_transcoder, activation_store, model, n_training_steps)
                sparse_transcoder.train()"""
                
            pbar.set_description(
                f"{n_training_steps}| Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        #sparse_transcoder1.remove_gradient_parallel_to_decoder_directions()
        #sparse_transcoder2.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


        """# checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = sparse_transcoder1.cfg
            path1 = f"{sparse_transcoder1.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder1.get_name()}.pt"
            path2 = f"{sparse_transcoder2.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder2.get_name()}.pt"
            #log_feature_sparsity_path = f"{sparse_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder.get_name()}_log_feature_sparsity.pt"
            sparse_transcoder1.save_model(path1)
            sparse_transcoder2.save_model(path2)
            #torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0"""
        
                
            
        n_training_steps += 1
    """
    log_feature_sparsity_path = f"{sparse_transcoder.cfg.checkpoint_path}/final_{sparse_transcoder.get_name()}_log_feature_sparsity.pt"
    sparse_transcoder.save_model(path)
    torch.save(log_feature_sparsity, log_feature_sparsity_path)
    if cfg.log_to_wandb:
        sparsity_artifact = wandb.Artifact(
                f"{sparse_transcoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
            )
        sparsity_artifact.add_file(log_feature_sparsity_path)
        wandb.log_artifact(sparsity_artifact)"""
        

    return mask

