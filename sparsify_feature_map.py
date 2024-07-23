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

def apply_causal_mask(attn_scores):
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)).cuda(), diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, -1e9)
        return attn_scores
    
def train_mask(
    cfg,
    model,
    query_transcoder,
    key_transcoder,
    activation_store
):
    print("TRAIN STARTED")
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.batch_size
    n_training_steps = 0
    n_training_tokens = 0

    if cfg.log_to_wandb:
        wandb.init(entity=cfg.entity, project=cfg.wandb_project, config=cfg, name=cfg.run_name)
    
    #define initial feature-map
    q_features = einops.rearrange(query_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = query_transcoder.d_head)
    k_features = einops.rearrange(key_transcoder.W_dec, "d_hidden (n_head d_head) -> d_hidden n_head d_head", d_head = key_transcoder.d_head)
    feature_map_base = einops.einsum(q_features, k_features, "d_hidden_Q n_head d_head, d_hidden_K n_head d_head -> d_hidden_Q d_hidden_K n_head")

    mask = nn.Parameter(torch.ones(query_transcoder.d_hidden, key_transcoder.d_hidden, query_transcoder.n_head).cuda())
    
    optimizer = Adam([mask], lr = cfg.lr)
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=cfg.lr / 10, # heuristic for now. 
    )
    
    attn_scores_norm = cfg.d_head ** 0.5 if cfg.attn_scores_normed else 1
        

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        
        data = activation_store.next_batch()
        data = einops.rearrange(data, "(batch posn) d_model -> batch posn d_model", posn = cfg.context_size)
        
        #Ground truth activations for losses
        true_queries = einops.einsum(model.W_Q[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[query_transcoder.layer]
        true_keys = einops.einsum(model.W_K[query_transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[query_transcoder.layer]
        true_queries_flat = einops.rearrange(true_queries, " ... n_head d_head -> ... (n_head d_head)")
        true_keys_flat = einops.rearrange(true_keys, " ... n_head d_head -> ... (n_head d_head)")
        true_scores = einops.einsum(true_queries, true_keys, " ... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")/attn_scores_norm
        true_patt = apply_causal_mask(true_scores).log_softmax(-1)
        
        #mask feature-map
        feature_map = einops.einsum(q_features, k_features, "d_hidden_Q n_head d_head, d_hidden_K n_head d_head -> d_hidden_Q d_hidden_K n_head")
        feature_map = feature_map * mask
        
        #feature activations
        feature_acts_Q = F.relu(einops.einsum((true_queries_flat - query_transcoder.b_dec), query_transcoder.W_enc, "... d_model, d_model d_hidden -> ... d_hidden") + query_transcoder.b_enc)
        feature_acts_K = F.relu(einops.einsum((true_keys_flat - key_transcoder.b_dec), key_transcoder.W_enc, "... d_model, d_model d_hidden -> ... d_hidden") + key_transcoder.b_enc)
        #given feature acts and map between features, compute attention contribution from feature-pairs
        attn_contribution = einops.einsum(feature_acts_Q, feature_map, "batch posnQ d_hidden_Q, d_hidden_Q d_hidden_K n_head -> batch posnQ d_hidden_K n_head")

        attn_contribution = einops.einsum(attn_contribution, feature_acts_K, "batch posnQ d_hidden_K n_head, batch posnK d_hidden_K -> batch posnQ posnK n_head")
        
        #compute attention contribution from key-features to query-biase 
        bias_reshape = einops.rearrange(query_transcoder.b_dec, "(n_head d_head) -> n_head d_head", n_head = cfg.n_head)
        bias_acts = einops.einsum(k_features, bias_reshape, "d_hidden_K n_head d_head, n_head d_head -> n_head d_hidden_K")
        contr_from_bias = einops.einsum(bias_acts, feature_acts_K, "n_head d_hidden_K, ... d_hidden_K -> ... n_head").unsqueeze(1)
        #pattern and loss
        attn_scores_reconstr = (attn_contribution + contr_from_bias)/attn_scores_norm
        attn_scores_reconstr = einops.rearrange(attn_scores_reconstr, "batch posnQ posnK n_head -> batch n_head posnQ posnK")
        reconstr_patt = apply_causal_mask(attn_scores_reconstr).log_softmax(-1)
        true_patt_flat = true_patt.view((-1, true_patt.shape[-1]))
        reconstr_patt = reconstr_patt.view((-1, reconstr_patt.shape[-1]))
        kl = torch.nn.KLDivLoss(reduction = "batchmean", log_target = True)
        print(reconstr_patt.shape, true_patt_flat.shape)
        kl_div = kl(reconstr_patt, true_patt_flat)
        reg_loss = cfg.reg_coefficient * torch.sqrt(mask.float().abs() + cfg.eps).sum()
        loss = kl_div + reg_loss
        
        n_training_tokens += cfg.batch_size

        with torch.no_grad():
            frac_zeros = (mask == 0).float().mean()
            frac_below_half = (mask < 0.5).float().mean()
            frac_decreasing = (mask < 0.9).float().mean()
            zeros = (mask > 0).float()
            zero_rows = (mask.sum(-1).sum(-1) == 0).sum()
            zero_cols = (mask.sum(-1).sum(0) == 0).sum()
            
            wandb.log(
                    {
                        # losses
                        "losses/kl_div": kl_div.item(),
                        "losses/reg_loss": reg_loss.item(),
                        # variance explained
                        "metrics/frac_zeros" : frac_zeros.item(),
                        "metrics/frac_below_half": frac_below_half.item(),
                        "metrics/frac_decreasing": frac_decreasing.item(),
                        "metrics/zero_rows": zero_rows.item(),
                        "metrics/zero_cols": zero_cols.item()

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
            pbar.update(cfg.batch_size)


        optimizer.zero_grad()  # Ensure gradients are zeroed before backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
           mask.clamp_(0, 1)
            
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

