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


from transcoder_training_parallel import apply_causal_mask, compute_ground_truth, flatten_heads, unflatten_heads, kl_loss_scores


    
def sparsity_transcoder(
    cfg,
    model: HookedTransformer,
    query_transcoder: SparseTranscoder,
    key_transcoder: SparseTranscoder,
    activation_store: ActivationsStore,
):
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.train_batch_size
    
    if cfg.log_to_wandb:
        wandb.init(entity=cfg.entity, project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # define mask
    mask = nn.Parameter(torch.ones(query_transcoder.d_hidden, key_transcoder.d_hidden, cfg.n_head).to(cfg.device))
    
    optimizer = Adam([mask], lr = cfg.lr)
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=cfg.lr / 10, # heuristic for now. 
    )
    
    with torch.no_grad():
        # define initial feature-map - TODO: assuming decoder is normalised!
        q_decoder = einops.rearrange(query_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = query_transcoder.d_head)
        k_decoder = einops.rearrange(key_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = key_transcoder.d_head)
        qk_feature_map = einops.einsum(q_decoder, k_decoder, "d_hidden_Q n_head d_head, d_hidden_K n_head d_head -> d_hidden_Q d_hidden_K n_head")
        
        # define bias part of overall scores transcoder
        key_decoder_unflat = einops.rearrange(key_transcoder.W_dec, "d_hidden_K (n_head d_head) -> d_hidden_K n_head d_head", d_head=key_transcoder.d_head)
        query_bias_unflat = einops.rearrange(query_transcoder.b_dec_out, "(n_head d_head) -> n_head d_head", d_head=query_transcoder.d_head)
        linear_key_feat_weights = einops.einsum(query_bias_unflat, key_decoder_unflat, "n_head d_head, d_hidden_K n_head d_head -> n_head d_hidden_K")
    
    with tqdm(total=total_training_tokens, desc="Training Mask") as pbar:
        for step in range(total_training_steps):

            # Do a training step.
            optimizer.zero_grad()
            
            # Get data and true values to match.
            resid = activation_store.next_batch()
            resid = einops.rearrange(resid, "(batch posn) d_model -> batch posn d_model", posn=cfg.context_size)
            true_queries, true_keys, _, true_patt = compute_ground_truth(model, resid, cfg, cfg.attn_scores_norm)
            
            # Forward transcoder passes for features
            if cfg.as_sae:
                key_trans_in = flatten_heads(true_keys)
                query_trans_in = flatten_heads(true_queries)
            else:
                key_trans_in, query_trans_in = resid, resid
            _, feature_actsQ, _ = query_transcoder(query_trans_in)
            _, feature_actsK, _ = key_transcoder(key_trans_in)
            
            feature_actsK = feature_actsK.detach()
            feature_actsQ = feature_actsQ.detach()
            
            
            # Reconcstruct attention scores from masked QK feature map
            # feature_acts - both batch x posn x d_hidden
            masked_qk_feature_map = qk_feature_map * mask
            bilinear_term = einops.einsum(feature_actsQ, masked_qk_feature_map, "batch posn_q d_hidden_Q, d_hidden_Q d_hidden_K n_head -> batch posn_q d_hidden_K n_head")
            bilinear_term = einops.einsum(bilinear_term, feature_actsK, "batch posn_q d_hidden_K n_head, batch posn_k d_hidden_K -> batch n_head posn_q posn_k")
            
            # linear_key_feat_weights is fixed during this training so defined outside loop.
            contr_from_bias = einops.einsum(linear_key_feat_weights, feature_actsK, "n_head d_hidden_K, batch posn_k d_hidden_K -> batch n_head posn_k")
            contr_from_bias = einops.repeat(contr_from_bias, "batch n_head posn_k -> batch n_head posn_q posn_k", posn_q=cfg.context_size)
            attn_scores_reconstr = bilinear_term + contr_from_bias
            
            
            # Loss.
            patt_loss = kl_loss_scores(attn_scores_reconstr, true_patt, cfg.attn_scores_norm)
            mask_sparsity_reg = torch.sqrt(mask.float().abs() + cfg.eps).sum()
            loss = patt_loss + cfg.mask_reg_coeff * mask_sparsity_reg
            

            # Train step.
            loss.backward()
            scheduler.step()
            optimizer.step()
            with torch.no_grad():
                mask = torch.clamp_(mask, min = 0, max = 1)    # make sure to keep mask between 0 and 1

            # Tracking and logging things.
            pbar.update(cfg.train_batch_size)
            pbar.set_postfix({"Patt Loss": f"{patt_loss.item():.3f}", "Sparsity": f"{mask_sparsity_reg.item():.3f}", "Fraction Zeros": f"{(mask == 0).float().mean().item():.3f}"})


            if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
                with torch.no_grad():

                    frac_zeros = (mask == 0).float().mean()
                    frac_below_half = (mask < 0.5).float().mean()
                    frac_decreasing = (mask < 0.9).float().mean()
                    zero_rows = (mask.sum(-1).sum(-1) == 0).sum()
                    zero_cols = (mask.sum(-1).sum(0) == 0).sum()
                    
                    wandb.log(
                            {
                                # losses
                                "losses/kl_div": patt_loss.item(),
                                "losses/reg_loss": mask_sparsity_reg.item(),
                                # variance explained
                                "metrics/frac_zeros" : frac_zeros.item(),
                                "metrics/frac_below_half": frac_below_half.item(),
                                "metrics/frac_decreasing": frac_decreasing.item(),
                                "metrics/zero_rows": zero_rows.item(),
                                "metrics/zero_cols": zero_cols.item()
                            },
                            step=step,
                        )


    if cfg.log_to_wandb:
        wandb.finish()

    return mask

