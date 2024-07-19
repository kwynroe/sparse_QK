import torch
import wandb


class SparsityLogger:
    "This needs to be logged more often to keep a true moving average."
    # TODO: change to real EMA?

    @torch.no_grad() 
    def __init__(self, cfg) -> None:
        self.act_freq_scores_q = torch.zeros(cfg.d_hidden, device=cfg.device)
        self.act_freq_scores_k = torch.zeros(cfg.d_hidden, device=cfg.device)

    def update(self, feature_actsQ, feature_actsK):
        self.act_freq_scores_q += (feature_actsQ.abs() > 0).float().sum(0).sum(0)
        self.act_freq_scores_k += (feature_actsK.abs() > 0).float().sum(0).sum(0)
    
    def log_to_wandb(self, n_training_tokens, step):
        feature_sparsity_q = self.act_freq_scores_q / n_training_tokens
        feature_sparsity_k = self.act_freq_scores_k / n_training_tokens
        wandb.log(
            {
                "sparsity/below_1e-5_Q": (feature_sparsity_q < 1e-5).float().mean().item(),
                "sparsity/above_1e-1_K": (feature_sparsity_k > 1e-1).float().mean().item(),
                "sparsity/above_1e-1_Q": (feature_sparsity_q > 1e-1).float().mean().item(),
                "sparsity/below_1e-5_K": (feature_sparsity_k < 1e-5).float().mean().item(),
                "sparsity/avg_log_freq_Q": (torch.log10(feature_sparsity_q).mean()).float().mean().item(),
                "sparsity/avg_log_freq_K": (torch.log10(feature_sparsity_k).mean()).float().mean().item(),
            },
            step=step,
        )


class WandbLogger:

    @torch.no_grad() 
    def __init__(self, cfg, optimizer) -> None:
        self.cfg = cfg
        self.n_frac_active_tokens = 0     # TODO: is this just no tokens used again??
        self.optimizer = optimizer
        
    def log_to_wandb(
            self,
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
            attn_scores_loss_full_pred,
            attn_score_loss_true_keys,
            attn_score_loss_true_queries,
            true_patt_flat,
            patt_full_reconstr,
            step,
        ):
        #Calculate average max pattern error and fraction of contexts where reconstruction correctly identified most interesting source token
        #Not very rigorous but helpful to plot while training!
        patt_max_diff = torch.max(torch.abs((torch.exp(true_patt_flat) - torch.exp(patt_full_reconstr))), dim = -1).values.mean()
        frac_accurate = (torch.argmax(patt_full_reconstr, dim = -1) == torch.argmax(true_patt_flat, dim = -1)).float().mean()
        
        # Calculate the sparsities, and add it to a list, calculate sparsity metrics
        l0_Q = (feature_actsQ > 0).float().sum(-1).mean()
        l0_K = (feature_actsK > 0).float().sum(-1).mean()
            
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
                
                #misc details
                "details/n_training_tokens": step * self.cfg.train_batch_size,
                "details/pred_key_mean": reconstr_keys.mean().item(),
                "details/pred_query_mean": reconstr_queries.mean().item(),
                "details/patt_max_diff": patt_max_diff.item(),
                "details/frac_acc": frac_accurate.item(),
                "details/lr": self.optimizer.param_groups[0]["lr"],

            },
            step=step,
        )