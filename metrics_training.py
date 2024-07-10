import torch
import wandb


class WandbLogger:

    def __init__(self, cfg, query_transcoder, key_transcoder, optimizer) -> None:
        self.cfg = cfg
        self.act_freq_scores_q = torch.zeros(query_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
        self.act_freq_scores_k = torch.zeros(key_transcoder.cfg.d_hidden, device=query_transcoder.cfg.device)
        self.n_frac_active_tokens = 0     # TODO: is this just no tokens used again??
        self.optimizer = optimizer
        
    @torch.no_grad() 
    def log_to_wandb(
            self,
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
        ):
        #Calculate average max pattern error and fraction of contexts where reconstruction correctly identified most interesting source token
        #Not very rigorous but helpful to plot while training!
        patt_max_diff = torch.max(torch.abs((torch.exp(true_patt_flat) - torch.exp(patt_full_reconstr))), dim = -1).values.mean()
        frac_accurate = (torch.argmax(patt_full_reconstr, dim = -1) == torch.argmax(true_patt_flat, dim = -1)).float().mean()
        
        # Calculate the sparsities, and add it to a list, calculate sparsity metrics
        self.act_freq_scores_q += (feature_actsQ.abs() > 0).float().sum(0).sum(0)
        self.act_freq_scores_k += (feature_actsK.abs() > 0).float().sum(0).sum(0)
        feature_sparsity_q = self.act_freq_scores_q / n_training_tokens     # TODO: this is changed - check
        feature_sparsity_k = self.act_freq_scores_k / n_training_tokens
        l0_Q = (feature_actsQ > 0).float().sum(-1).mean()
        l0_K = (feature_actsK > 0).float().sum(-1).mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]
            
        #per_token_l2_loss = (transcoder_out - target).pow(2).sum(dim=-1).squeeze()
        total_variance = (true_scores - true_scores.mean()).pow(2).mean()
        
        resid_var_q = (true_scores - pred_attn_scores_true_keys).pow(2).mean()
        resid_var_k = (true_scores - pred_attn_scores_true_queries).pow(2).mean()
        explained_var_q = 1 - resid_var_q / total_variance
        explained_var_k = 1 - resid_var_k / total_variance
        
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
                    "metrics/full_pred_diff": attn_scores_loss_full_pred.item(),  # TODO: this is without causal mask, so a weird thing to track??
                    "metrics/loss_true_keys": attn_score_loss_true_keys.item(),
                    "metrics/loss_true_queries": attn_score_loss_true_queries.item(),
                    "metrics/l0_Q": l0_Q.item(),
                    "metrics/l0_K": l0_K.item(),

                    # sparsity
                    "sparsity/below_1e-5_Q": (feature_sparsity_q < 1e-5)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/above_1e-1_K": (feature_sparsity_k > 1e-1)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/above_1e-1_Q": (feature_sparsity_q > 1e-1)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/below_1e-5_K": (feature_sparsity_k < 1e-5)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/avg_log_freq_Q": (torch.log10(feature_sparsity_q).mean())
                    .float()
                    .mean()
                    .item(),
                    "sparsity/avg_log_freq_K": (torch.log10(feature_sparsity_k).mean())
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