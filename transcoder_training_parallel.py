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
from metrics_training import WandbLogger, SparsityLogger

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
    print("Initialising...")
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.train_batch_size

    if cfg.n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // cfg.n_checkpoints))[1:]
    
    optimizer = Adam(
        list(query_transcoder.parameters()) + list(key_transcoder.parameters()),
        lr=cfg.lr
    )
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=cfg.lr_warm_up_steps, 
        training_steps=total_training_tokens // cfg.train_batch_size,
        lr_end=cfg.lr / 10, # heuristic for now. 
    )

    sparsity_logger = SparsityLogger(cfg)
    wandb_logger = WandbLogger(cfg, optimizer)

    with tqdm(total=total_training_tokens, desc="Training Transcoders") as pbar:
        for step in range(total_training_steps):
            
            # Do a training step.
            optimizer.zero_grad()
            query_transcoder.train()  # TODO: think these might be redundant?
            key_transcoder.train()

            # Make sure the W_dec is still zero-norm
            query_transcoder.set_decoder_norm_to_unit_norm()
            key_transcoder.set_decoder_norm_to_unit_norm()
            
            data = activation_store.next_batch()
            data = einops.rearrange(data, "(batch posn) d_model -> batch posn d_model", posn = cfg.context_size)
            
            true_queries, true_keys, true_scores, true_patt_view = compute_ground_truth(model, data, cfg, cfg.attn_scores_norm)
            
            # Forward transcoder passes.
            reconstr_queries_flat, feature_actsQ, mse_lossQ, reg_lossQ = query_transcoder(
                data,
                flatten_heads(true_queries)    # TODO: I don't think mse and target should be in here, should use feature_acts instead
            )
            reconstr_queries = unflatten_heads(reconstr_queries_flat, cfg.n_head)
            
            reconstr_keys_flat, feature_actsK,  mse_lossK, reg_lossK = key_transcoder(
                data,
                flatten_heads(true_keys)
            )
            reconstr_keys = unflatten_heads(reconstr_keys_flat, cfg.n_head)
            
            #Calculate attention scores using reconstructed components (full reconstructed + half reconstructed)
            pred_scores_true_keys = einops.einsum(reconstr_queries, true_keys, " ... posnq n_head d_head, ... posnk n_head d_head -> ... n_head posnq posnk")
            pred_scores_true_queries = einops.einsum(reconstr_keys, true_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")
            pred_scores_full = einops.einsum(reconstr_keys, reconstr_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")
            
            # Losses with full + half reconstructions
            patt_loss_true_queries = kl_loss_scores(pred_scores_true_queries, true_patt_view, cfg.attn_scores_norm)
            patt_loss_true_keys = kl_loss_scores(pred_scores_true_keys, true_patt_view, cfg.attn_scores_norm)
            patt_loss_full_pred = kl_loss_scores(pred_scores_full, true_patt_view, cfg.attn_scores_norm)
            loss = 3 * patt_loss_full_pred + patt_loss_true_queries + patt_loss_true_keys + reg_lossQ + reg_lossK
            
            # Train step.
            loss.backward()
            optimizer.step()
            scheduler.step()
            

            # Tracking and logging things.
            pbar.update(cfg.train_batch_size)
            pbar.set_postfix({"Loss": f"{loss.item():.3f}"})
            sparsity_logger.update(feature_actsQ, feature_actsK)  # update this every step.

            if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
                sparsity_logger.log_to_wandb(n_training_tokens, step)
                wandb_logger.log_to_wandb(
                    feature_actsQ,
                    feature_actsK,
                    mse_lossQ,
                    mse_lossK,
                    reg_lossQ,
                    reg_lossK,
                    reconstr_keys,
                    reconstr_queries,
                    patt_loss_true_keys,
                    patt_loss_true_queries,
                    patt_loss_full_pred,
                    ((pred_scores_full) - (true_scores)).pow(2).mean(),                # attn_scores_loss_full_pred
                    ((pred_scores_true_keys) - (true_scores)).pow(2).mean(),           # attn_score_loss_true_keys
                    ((pred_scores_true_queries) - (true_scores)).pow(2).mean(),        # attn_score_loss_true_queries
                    true_patt_view,
                    flat_pattern_from_scores(pred_scores_full, cfg.attn_scores_norm),  # patt_full_reconstr
                    step,
                )

            n_training_tokens = step * cfg.train_batch_size
            if cfg.n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
                path_q = f"{cfg.checkpoint_path}/{n_training_tokens}_{query_transcoder.get_name()}.pt"
                path_k = f"{cfg.checkpoint_path}/{n_training_tokens}_{key_transcoder.get_name()}.pt"
                query_transcoder.save_model(path_q)
                key_transcoder.save_model(path_k)
                checkpoint_thresholds.pop(0)
                if len(checkpoint_thresholds) == 0:
                    cfg.n_checkpoints = 0        

    return query_transcoder, key_transcoder




def compute_ground_truth(model, data, cfg, attn_scores_norm):
    """
    Compute ground truth queries, keys, scores, and attention patterns.

    Args:
        model (torch.nn.Module): The main model being trained.
        data (torch.Tensor): Input data tensor.
        cfg (Config): Configuration object containing model parameters.
        attn_scores_norm: Either 1 or 1/sqrt(d_head)

    Returns:
        tuple: Contains true_queries, true_keys, true_scores, and true_patt tensors.
    """
    true_queries = einops.einsum(model.W_Q[cfg.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[cfg.layer]
    true_keys = einops.einsum(model.W_K[cfg.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[cfg.layer]
    true_scores = einops.einsum(true_queries, true_keys, "... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")
    true_patt_view = flat_pattern_from_scores(true_scores, attn_scores_norm)
    return true_queries, true_keys, true_scores, true_patt_view


def flatten_heads(tensor):
    return einops.rearrange(tensor, " ... n_head d_head -> ... (n_head d_head)")

def unflatten_heads(tensor, n_head):
    return einops.rearrange(tensor, " ... (n_head d_head) -> ... n_head d_head", n_head=n_head)


def flat_pattern_from_scores(scores, attn_scores_norm):
    pattern = apply_causal_mask(scores / attn_scores_norm).log_softmax(-1)
    flat_pattern = pattern.view((-1, pattern.shape[-1]))
    return flat_pattern


def kl_loss_scores(predicted_scores, true_patt_view, attn_scores_norm):
    "Takes in non-flattened scores and pattern and gives kl loss."
    pred_patt_flat = flat_pattern_from_scores(predicted_scores, attn_scores_norm)
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target = True)  # TODO could probably use scores instead??
    return kl_loss(pred_patt_flat, true_patt_view)