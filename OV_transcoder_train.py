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
        attn_scores.masked_fill_(mask, -1e9)
        return attn_scores


def train_OV_transcoder(
    cfg,
    model: HookedTransformer,
    transcoder,
    key_transcoder,
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

    total_training_tokens = transcoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores1 = torch.zeros(transcoder.cfg.d_hidden, device=transcoder.cfg.device)
    n_forward_passes_since_fired1 = torch.zeros(transcoder.cfg.d_hidden, device=transcoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(list(transcoder.parameters()),
                     lr = transcoder.cfg.lr)
    print("gonna schedule!")
    scheduler = get_scheduler(
        transcoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = transcoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=transcoder.cfg.lr / 10, # heuristic for now. 
    )
    transcoder.initialize_b_dec(activation_store)
    transcoder.train()
    
    
    if cfg.attn_scores_norm:
        attn_scores_norm = transcoder.d_head ** 0.5
    else:
        attn_scores_norm = 1
    print("gonna progress bar!")
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        
        # Do a training step.
        transcoder.train()
        # Make sure the W_dec is still zero-norm
        transcoder.set_decoder_norm_to_unit_norm()
           
        scheduler.step()
        optimizer.zero_grad()
        
        data = activation_store.next_batch()
        data = einops.rearrange(data, "(batch posn) d_model -> batch posn d_model", posn = cfg.context_size)
        V = einops.einsum(model.W_V[transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_V[transcoder.layer]
        post = einops.einsum(V, model.W_O[transcoder.layer], "... n_head d_head, n_head d_head d_model -> ... n_head d_model") + model.b_O[transcoder.layer] 
        
        
        #ground truth activations for loss
        true_keys = einops.einsum(model.W_K[transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[transcoder.layer]
        true_queries = einops.einsum(model.W_Q[transcoder.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[transcoder.layer]
        true_scores = einops.einsum(true_queries, true_keys, "batch posnQ n_head d_head, batch posnK n_head d_head -> batch n_head posnQ posnK")/attn_scores_norm
        true_patt = apply_causal_mask(true_scores).softmax(-1)
        attn_out = einops.einsum(post, true_patt, "batch posnK n_head d_model, batch n_head posnQ posnK -> batch posnQ n_head d_model")
        attn_out = einops.rearrange(attn_out, "... n_head d_head -> ... (n_head d_head)")
        _, feature_actsK, _ = key_transcoder(data)
        
        # Forward and Backward Passes
        reconstr_post, feature_acts, reg_loss, gamma_reg_loss = transcoder(
            data,
            feature_actsK,
            true_patt
        )
        
        mse_loss = (reconstr_post - attn_out).pow(2).sum(-1).mean()
        
    
        #Full loss
        loss = mse_loss + cfg.reg_coefficient*reg_loss + cfg.gamma_reg_coefficient*gamma_reg_loss
        
        #Feature Sparsity Calculations
        did_fireQ = ((feature_acts > 0).float().sum(0).sum(0) > 0)
        n_forward_passes_since_fired1 += 1
        n_forward_passes_since_fired1[did_fireQ] = 0

        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores1 += (feature_acts.abs() > 0).float().sum(0).sum(0)

            n_frac_active_tokens += batch_size
            feature_sparsity1 = act_freq_scores1 / n_frac_active_tokens

            l0 = (feature_acts > 0).float().sum(-1).mean()
            current_learning_rate = optimizer.param_groups[0]["lr"]
                
            #per_token_l2_loss = (transcoder_out - target).pow(2).sum(dim=-1).squeeze()
            total_variance = (attn_out - attn_out.mean()).pow(2).mean()
            
            resid_var = (attn_out - reconstr_post).pow(2).mean()
            explained_var = 1 - resid_var / total_variance

            
            if use_wandb:
                if (n_training_steps + 1) % wandb_log_frequency == 0:
                    wandb.log(
                            {
                                # losses
                                "losses/mse_loss": mse_loss.item(),
                
                                "losses/reg_loss": reg_loss.item(),
                                
                                "losses/gamma_reg_loss": gamma_reg_loss.item(),
                     
                
                                # metrics
       
                                "metrics/l0_Q": l0.item(),
                                "metrics/explained_var": explained_var.item(),
                                "metrics/gamma_zeros": (transcoder.gamma == 0).float().mean().item(),
                                "metrics/gamma_decreasing":(transcoder.gamma < 0.9).float().mean().item(),

                                # sparsity
                                "sparsity/below_1e-5": (feature_sparsity1 < 1e-5)
                                .float()
                                .mean()
                                .item(),
                                "sparsity/above_1e-1": (feature_sparsity1 > 1e-1)
                                .float()
                                .mean()
                                .item(),

                                "sparsity/avg_log_freq": (torch.log10(feature_sparsity1).mean())
                                .float()
                                .mean()
                                .item(),
                                
                                #misc details
                                "details/n_training_tokens": n_training_tokens,
                                "details/lr": current_learning_rate

                            },
                            step=n_training_steps,
                    )
            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        #transcoder.remove_gradient_parallel_to_decoder_directions()
        #key_transcoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        
        with torch.no_grad():
           transcoder.gamma.clamp_(0, 1)


        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = transcoder.cfg
            path1 = f"{transcoder.cfg.checkpoint_path}/{n_training_tokens}_{transcoder.get_name()}.pt"

            #log_feature_sparsity_path = f"{sparse_transcoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_transcoder.get_name()}_log_feature_sparsity.pt"
            transcoder.save_model(path1)

            #torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0        
        n_training_steps += 1

    return transcoder
