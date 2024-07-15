
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

def apply_causal_mask(attn_scores):

        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1)).cuda(), diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, -1e9)
        return attn_scores
    

def train_OV_circuit(model, 
    key_transcoder, 
    cfg, 
    activation_store,
    use_wandb: bool = True,
    wandb_log_frequency: int = 10):
    
    batch_size = cfg.batch_size
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    
    #Initisialise maps
    kv_map = torch.ones(key_transcoder.d_hidden, cfg.n_features).to(cfg.device)
    features_out = torch.nn.init.kaiming_uniform_(
                torch.empty(key_transcoder.d_hidden, cfg.n_heads, cfg.d_model, dtype=cfg.dtype, device=cfg.device)
            )
    features_out_bias = torch.zeros(cfg.n_head, cfg.d_model).cuda()
    
    optimizer = Adam(list(kv_map) + list(features_out) + list(features_out_bias),
                     lr = cfg.lr)

    if cfg.attn_scores_norm:
        attn_scores_norm = cfg.d_head**0.5
    else:
        attn_scores_norm = 1
        
    print("gonna schedule!")
    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=cfg.lr / 10,) # heuristic for now. 
        
  
    n_training_steps = 0
    n_training_tokens = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    
    
    print("gonna progress bar!")
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        mask = torch.clip(kv_map, min = 0, max = 1)
        # Do a training step.
        # Make sure the W_dec is still zero-norm
        #transcoder.set_decoder_norm_to_unit_norm()
           
        scheduler.step()
        optimizer.zero_grad()
        
        data = activation_store.next_batch()
        data = einops.rearrange(data, "batch posn d_model -> batch posn d_model", posn = cfg.context_size)
        
        V = einops.einsum(model.W_V[cfg.layer], data, "d_model n_head d_head, ... d_model -> ... n_head d_head") + model.b_V[cfg.layer]
        post = einops.einsum(V, model.W_O[cfg.layer], "... n_head d_head, n_head d_head d_model -> ... n_head d_model") + model.b_O[cfg.layer] 
        
        
        true_queries = einops.einsum(model.W_Q[cfg.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[cfg.layer]
        true_keys = einops.einsum(model.W_K[cfg.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[cfg.layer]
        true_scores = einops.einsum(true_queries, true_keys, " ... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")/attn_scores_norm
        true_patt = apply_causal_mask(true_scores).log_softmax(-1)
        
        #ground truth activations for loss

        # Forward and Backward Passes
        _, feature_acts, _, reg_loss = key_transcoder(
            data,
            post
        )
        #Feature Sparsity Calculations
        did_fireQ = ((feature_acts > 0).float().sum(0).sum(0) > 0)
        n_forward_passes_since_fired1 += 1
        n_forward_passes_since_fired1[did_fireQ] = 0

        n_training_tokens += batch_size
        
        #Losses
        value_acts = einops.einsum(feature_acts, kv_map, "... d_hiddenK, d_hiddenK d_hiddenV -> ... d_hiddenV")
        value_acts = einops.einsum(value_acts, true_patt, "batch posnK d_hiddenV, batch n_head posnQ posnK -> batch n_head posnQ d_hiddenV")
        head_outs = einops.einsum(value_acts, features_out, "batch n_head posnQ d_hiddenV, d_hiddenV n_head d_model -> batch posnQ n_head d_model") + features_out_bias
        mse_loss = (head_outs - post).pow(2).mean()
        
        reg_coeff = (n_training_tokens / total_training_tokens) * cfg.max_reg_coeff
        reg_loss = reg_coeff * torch.sqrt(kv_map.float().abs() + cfg.eps).sum()
        #Full loss
        loss = mse_loss + reg_loss

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores1 += (feature_acts.abs() > 0).float().sum(0).sum(0)

            n_frac_active_tokens += batch_size
            feature_sparsity1 = act_freq_scores1 / n_frac_active_tokens

            l0 = (feature_acts > 0).float().sum(-1).mean()
            current_learning_rate = optimizer.param_groups[0]["lr"]
                
            #per_token_l2_loss = (transcoder_out - target).pow(2).sum(dim=-1).squeeze()
            total_variance = (post - post.mean()).pow(2).mean()
            
            resid_var = (post - head_outs).pow(2).mean()
            explained_var = 1 - resid_var / total_variance
            frac_zeros = (kv_map == 0).mean()
            frac_decreasing = (kv_map < 0.9).mean()
            frac_below_half =  (kv_map < 0.5).mean()
            
            if use_wandb:
                if (n_training_steps + 1) % wandb_log_frequency == 0:
                    wandb.log(
                            {
                                # losses
                                "losses/mse_loss": mse_loss.item(),
                                "losses/reg_loss": reg_loss.item(),
                     
                
                                # metrics
       
                                "metrics/l0_Q": l0.item(),
                                "metrics/explained_var": explained_var.item(),

                                # sparsity
                                "sparsity/frac_zero": frac_zeros.item(),
                                "sparsity/below_half": frac_below_half.item(),
                                "sparsity/frac_decreasing": frac_decreasing.item(),
                                
                                
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
