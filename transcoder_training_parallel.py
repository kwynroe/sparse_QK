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
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_method = None,
    use_wandb: bool = True,
    wandb_log_frequency: int = 1,
):
    print("TRAIN STARTED")
    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()

    total_training_tokens = query_transcoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores1 = torch.zeros(query_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
    act_freq_scores2 = torch.zeros(key_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
    n_forward_passes_since_fired1 = torch.zeros(query_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
    n_forward_passes_since_fired2 = torch.zeros(key_transcoder.cfg.d_hidden, device=key_transcoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(list(query_transcoder.parameters()) + list(key_transcoder.parameters()),
                     lr = query_transcoder.cfg.lr)
    print("gonna schedule!")
    scheduler = get_scheduler(
        query_transcoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = query_transcoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=query_transcoder.cfg.lr / 10, # heuristic for now. 
    )
    query_transcoder.initialize_b_dec(activation_store)
    key_transcoder.initialize_b_dec(activation_store)
    query_transcoder.train()
    key_transcoder.train()
    
    if query_transcoder.cfg.attn_scores_normed:
        attn_scores_norm = query_transcoder.d_head**0.5
    else:
        attn_scores_norm = 1
    print("gonna progress bar!")
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        
        # Do a training step.
        query_transcoder.train()
        key_transcoder.train()
        # Make sure the W_dec is still zero-norm
        query_transcoder.set_decoder_norm_to_unit_norm()
        key_transcoder.set_decoder_norm_to_unit_norm()
           
        scheduler.step()
        optimizer.zero_grad()
        
        ghost_grad_neuron_mask1 = (n_forward_passes_since_fired1 > query_transcoder.cfg.dead_feature_window).bool()
        ghost_grad_neuron_mask2 = (n_forward_passes_since_fired2 > key_transcoder.cfg.dead_feature_window).bool()
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
        reconstr_queries, feature_actsQ, lossQ, mse_lossQ, reg_lossQ, ghost_grad_lossQ = query_transcoder(
            data,
            true_queries_flatt,
            ghost_grad_neuron_mask1,
        )
        
        reconstr_keys, feature_actsK, lossK, mse_lossK, reg_lossK, ghost_grad_lossK = key_transcoder(
            data,
            true_keys_flatt,
            ghost_grad_neuron_mask2,
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
        
        #Calculate average max pattern error and fraction of contexts where reconstruction correctly identified most interesting source token
        #Not very rigorous but helpful to plot while training!
        patt_max_diff = torch.max(torch.abs((torch.exp(true_patt_flat) - torch.exp(patt_full_reconstr))), dim = -1).values.mean()
        frac_accurate = (torch.argmax(patt_full_reconstr, dim = -1) == torch.argmax(true_patt_flat, dim = -1)).float().mean()
        
        #Full loss
        loss = 3*patt_loss_full_pred + patt_loss_true_queries + patt_loss_true_keys + reg_lossQ + reg_lossK + ghost_grad_lossQ + ghost_grad_lossK
        
        #Feature Sparsity Calculations
        did_fireQ = ((feature_actsQ > 0).float().sum(0).sum(0) > 0)
        did_fireK = ((feature_actsK > 0).float().sum(0).sum(0) > 0)
        n_forward_passes_since_fired1 += 1
        n_forward_passes_since_fired1[did_fireQ] = 0
        n_forward_passes_since_fired2 += 1
        n_forward_passes_since_fired2[did_fireK] = 0
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores1 += (feature_actsQ.abs() > 0).float().sum(0).sum(0)
            act_freq_scores2 += (feature_actsK.abs() > 0).float().sum(0).sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity1 = act_freq_scores1 / n_frac_active_tokens
            feature_sparsity2 = act_freq_scores2 / n_frac_active_tokens
            l0_Q = (feature_actsQ > 0).float().sum(-1).mean()
            l0_K = (feature_actsK > 0).float().sum(-1).mean()
            current_learning_rate = optimizer.param_groups[0]["lr"]
                
            #per_token_l2_loss = (transcoder_out - target).pow(2).sum(dim=-1).squeeze()
            total_variance = (true_scores - true_scores.mean()).pow(2).mean()
            
            resid_var_q = (true_scores - pred_attn_scores_true_keys).pow(2).mean()
            resid_var_k = (true_scores - pred_attn_scores_true_queries).pow(2).mean()
            explained_var_q = 1 - resid_var_q / total_variance
            explained_var_k = 1 - resid_var_k / total_variance
            
            if use_wandb:
                if (n_training_steps + 1) % wandb_log_frequency == 0:
                    wandb.log(
                            {
                                # losses
                                "losses/mse_lossQ": mse_lossQ.item(),
                                "losses/mse_lossK": mse_lossK.item(),
                                "losses/reg_lossQ": reg_lossQ.item(),
                                "losses/reg_lossK": reg_lossK.item(),
                                "losses/patt_lossQ": patt_loss_true_keys.item(),
                                "losses/patt_lossK": patt_loss_true_queries.item(),
                                "losses/patt_loss_full": patt_loss_full_pred.item(),
                                
                                # metrics
                                "metrics/full_pred_diff": attn_scores_loss_full_pred.item(),
                                "metrics/loss_true_keys": attn_score_loss_true_keys.item(),
                                "metrics/loss_true_queries": attn_score_loss_true_queries.item(),
                                "metrics/l0_Q": l0_Q.item(),
                                "metrics/l0_K": l0_K.item(),

                                # sparsity
                                "sparsity/below_1e-5_Q": (feature_sparsity1 < 1e-5)
                                .float()
                                .mean()
                                .item(),
                                "sparsity/above_1e-1_K": (feature_sparsity1 > 1e-1)
                                .float()
                                .mean()
                                .item(),
                                "sparsity/above_1e-1_Q": (feature_sparsity2 > 1e-1)
                                .float()
                                .mean()
                                .item(),
                                "sparsity/below_1e-5_K": (feature_sparsity2 < 1e-5)
                                .float()
                                .mean()
                                .item(),
                                "sparsity/avg_log_freq_K": (torch.log10(feature_sparsity2).mean())
                                .float()
                                .mean()
                                .item(),
                                "sparsity/avg_log_freq_Q": (torch.log10(feature_sparsity1).mean())
                                .float()
                                .mean()
                                .item(),
                                
                                #misc details
                                "details/n_training_tokens": n_training_tokens,
                                "details/pred_key_mean": reconstr_keys.mean().item(),
                                "details/pred_query_mean": reconstr_queries.mean().item(),
                                "details/patt_max_diff": patt_max_diff.item(),
                                "details/frac_acc": frac_accurate.item(),
                                "details/lr": current_learning_rate

                            },
                            step=n_training_steps,
                    )
            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        #query_transcoder.remove_gradient_parallel_to_decoder_directions()
        #key_transcoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = query_transcoder.cfg
            path1 = f"{query_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{query_transcoder.get_name()}.pt"
            path2 = f"{key_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{key_transcoder.get_name()}.pt"
            #log_feature_sparsity_path = f"{sparse_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder.get_name()}_log_feature_sparsity.pt"
            query_transcoder.save_model(path1)
            key_transcoder.save_model(path2)
            #torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0        
        n_training_steps += 1

    return query_transcoder, key_transcoder

