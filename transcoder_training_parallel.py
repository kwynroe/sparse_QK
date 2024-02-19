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


def train_transcoder_on_language_model_parallel(
    cfg,
    model: HookedTransformer,
    sparse_transcoder1: SparseTranscoder,
    sparse_transcoder2: SparseTranscoder,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_method = None,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    dead_feature_window: int = 2000,  # how many training steps before a feature is considered dead
    use_wandb: bool = True,
    wandb_log_frequency: int = 1,
):
    print("TRAIN STARTED")
    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()

    total_training_tokens = sparse_transcoder1.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0
    steps_before_reset = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores1 = torch.zeros(sparse_transcoder1.cfg.d_hidden, device=sparse_transcoder1.cfg.device)
    act_freq_scores2 = torch.zeros(sparse_transcoder2.cfg.d_hidden, device=sparse_transcoder1.cfg.device)
    n_forward_passes_since_fired1 = torch.zeros(sparse_transcoder1.cfg.d_hidden, device=sparse_transcoder1.cfg.device)
    n_forward_passes_since_fired2 = torch.zeros(sparse_transcoder2.cfg.d_hidden, device=sparse_transcoder2.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(list(sparse_transcoder1.parameters()) + list(sparse_transcoder2.parameters()),
                     lr = sparse_transcoder1.cfg.lr)
    print("gonna schedule!")
    scheduler = get_scheduler(
        sparse_transcoder1.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = sparse_transcoder1.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=sparse_transcoder1.cfg.lr / 10, # heuristic for now. 
    )
    sparse_transcoder1.initialize_b_dec(activation_store)
    sparse_transcoder2.initialize_b_dec(activation_store)
    sparse_transcoder1.train()
    sparse_transcoder2.train()
    
    print("gonna progress bar!")
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        
        # Do a training step.
        sparse_transcoder1.train()
        sparse_transcoder2.train()
        # Make sure the W_dec is still zero-norm
        sparse_transcoder1.set_decoder_norm_to_unit_norm()
        sparse_transcoder2.set_decoder_norm_to_unit_norm()
           
        scheduler.step()
        optimizer.zero_grad()
        
        ghost_grad_neuron_mask1 = (n_forward_passes_since_fired1 > sparse_transcoder1.cfg.dead_feature_window).bool()
        ghost_grad_neuron_mask2 = (n_forward_passes_since_fired2 > sparse_transcoder2.cfg.dead_feature_window).bool()
        data = activation_store.next_batch()
        
        true_queries = einops.einsum(model.W_Q[sparse_transcoder1.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_Q[sparse_transcoder1.layer]
        true_keys = einops.einsum(model.W_K[sparse_transcoder1.layer], data, "n_head d_model d_head, ... d_model -> ... n_head d_head") + model.b_K[sparse_transcoder1.layer]
        true_scores = einops.einsum(true_queries, true_keys, " ... posn_q n_head d_head, ... posn_k n_head d_head -> ... n_head posn_q posn_k")/sparse_transcoder1.d_head**0.5

        true_queries_flatt = einops.rearrange(true_queries, " ... n_head d_head -> ... (n_head d_head)")
        true_keys_flatt = einops.rearrange(true_keys, " ... n_head d_head -> ... (n_head d_head)")
        # Forward and Backward Passes
        reconstr_queries, feature_actsQ, lossQ, mse_lossQ, reg_lossQ, ghost_grad_lossQ = sparse_transcoder1(
            data,
            true_queries_flatt,
            ghost_grad_neuron_mask1,
        )
        
        
        reconstr_keys, feature_actsK, lossK, mse_lossK, reg_lossK, ghost_grad_lossK = sparse_transcoder2(
            data,
            true_keys_flatt,
            ghost_grad_neuron_mask2,
        )
        
        reconstr_queries = einops.rearrange(reconstr_queries, " ... (n_head d_head) -> ... n_head d_head", n_head = 12)
        reconstr_keys = einops.rearrange(reconstr_keys, " ... (n_head d_head) -> ... n_head d_head", n_head = 12)
        
        pred_attn_scores_true_keys = einops.einsum(reconstr_queries, true_keys, " ... posnq n_head d_head, ... posnk n_head d_head -> ... n_head posnq posnk")/sparse_transcoder1.cfg.d_head**0.5
        pred_attn_scores_true_queries = einops.einsum(reconstr_keys, true_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")/sparse_transcoder1.cfg.d_head**0.5
        full_pred_attn_scores = einops.einsum(reconstr_keys, reconstr_queries, " ... posnk n_head d_head, ... posnq n_head d_head -> ... n_head posnq posnk")/sparse_transcoder1.cfg.d_head**0.5
        
        attn_score_loss_true_keys = ((pred_attn_scores_true_keys) - (true_scores)).pow(2).mean()
        attn_score_loss_true_queries = ((pred_attn_scores_true_queries) - (true_scores)).pow(2).mean()
        attn_scores_loss_full_pred = ((full_pred_attn_scores) - (true_scores)).pow(2).mean()
        patt_true_keys = pred_attn_scores_true_keys.softmax(-1)
        patt_true_queries = pred_attn_scores_true_queries.softmax(-1)
        true_patt = true_scores.softmax(-1)                                   
        #attn_score_loss_true_keys = torch.abs(torch.exp(pred_attn_scores_true_keys) - torch.exp(true_scores)).mean()
        #attn_score_loss_true_queries = torch.abs(torch.exp(pred_attn_scores_true_queries) - torch.exp(true_scores)).mean()
        attn_scores_loss_full_pred = torch.abs(torch.exp(full_pred_attn_scores) - torch.exp(true_scores)).mean()
        
        patt_loss_true_queries = (patt_true_queries - true_patt).pow(2).sum()
        patt_loss_true_keys = (patt_true_keys - true_patt).pow(2).sum()
        ghost_grad_scale_Q = (attn_score_loss_true_keys / (ghost_grad_lossQ + sparse_transcoder1.eps)).detach()
        ghost_grad_scale_K = (attn_score_loss_true_queries / (ghost_grad_lossK + sparse_transcoder1.eps)).detach()
        
        ghost_grad_lossQ = ghost_grad_lossQ * ghost_grad_scale_Q
        ghost_grad_lossK = ghost_grad_lossK * ghost_grad_scale_K
        
        #kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target = True)
        print(true_patt.shape)
        true_patt = true_scores.softmax(-1)
        full_pred_patt = full_pred_attn_scores.softmax(-1)
        true_patt = einops.rearrange(true_patt, "batch posnq posnk -> (batch posnq) posnk")
        full_pred_patt = einops.rearrange(full_pred_patt, "batch posnq posnk -> (batch posnq) posnk")
        #kl_div = kl_loss(true_patt, full_pred_patt)
        kl_div = (true_patt - full_pred_patt).pow(2).mean()
        patt_max_diff = torch.max(torch.abs((true_patt - full_pred_patt))).mean()
        
        #mse_lossQ + mse_lossK + 
        loss = attn_score_loss_true_keys + attn_score_loss_true_queries + patt_loss_true_queries + patt_loss_true_keys + reg_lossQ + reg_lossK + ghost_grad_lossQ + ghost_grad_lossK
        
        did_fireQ = ((feature_actsQ > 0).float().sum(0) > 0)
        did_fireK = ((feature_actsK > 0).float().sum(0) > 0)
        
        n_forward_passes_since_fired1 += 1
        n_forward_passes_since_fired1[did_fireQ] = 0
        n_forward_passes_since_fired2 += 1
        n_forward_passes_since_fired2[did_fireK] = 0
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores1 += (feature_actsQ.abs() > 0).float().sum(0)
            act_freq_scores2 += (feature_actsK.abs() > 0).float().sum(0)
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
            ##explained_variance = 1 - per_token_l2_loss/total_variance
                
            wandb.log(
                    {
                        # losses
                        "losses/mse_lossQ": mse_lossQ.item(),
                        "losses/mse_lossK": mse_lossQ.item(),
                        "losses/reg_lossQ": reg_lossQ.item(),
                        "losses/reg_lossK": reg_lossQ.item(),# normalize by reg coefficient
                        "losses/patt_lossQ": patt_loss_true_keys.item(),
                        "losses/patt_lossK": patt_loss_true_queries.item(),
                        "losses/ghost_grad_lossQ": ghost_grad_lossQ.item(),
                        "losses/ghost_grad_lossK": ghost_grad_lossK.item(),
                        # variance explained
                        "metrics/var_explained_Q" : explained_var_q.item(),
                        "metrics/var_explained_K" : explained_var_k.item(),
                        "metrics/full_pred_diff": attn_scores_loss_full_pred.item(),
                        "metrics/loss_true_keys": attn_score_loss_true_keys.item(),
                        "metrics/loss_true_queries": attn_score_loss_true_queries.item(),
                        "metrics/l0_Q": l0_Q.item(),
                        "metrics/l0_K": l0_K.item(),
                        "metrics/patt_kl": kl_div.item(),
                        #"metrics/score_var": total_variance.item(),
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
                        "sparsity/avg_log_freq_K": (torch.log(feature_sparsity2).mean())
                        .float()
                        .mean()
                        .item(),
                        "sparsity/avg_log_freq_Q": (torch.log(feature_sparsity1).mean())
                        .float()
                        .mean()
                        .item(),
                        

                        "details/n_training_tokens": n_training_tokens,
                        "details/pred_key_mean": reconstr_keys.mean().item(),
                        "details/pred_query_mean": reconstr_queries.mean().item(),
                        "details/patt_max_diff": patt_max_diff.item()

                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            """if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                sparse_transcoder.eval()
                run_evals(sparse_transcoder, activation_store, model, n_training_steps)
                sparse_transcoder.train()"""
                
            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        #sparse_transcoder1.remove_gradient_parallel_to_decoder_directions()
        #sparse_transcoder2.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


        # checkpoint if at checkpoint frequency
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
                n_checkpoints = 0
            """
            if cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_transcoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)
                
                sparsity_artifact = wandb.Artifact(
                    f"{sparse_transcoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)"""
                
            
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
        

    return sparse_transcoder1, sparse_transcoder2


@torch.no_grad()
def run_evals(sparse_autoencoder: SparseTranscoder, activation_store: ActivationsStore, model: HookedTransformer, n_training_steps: int):
    
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index
    
     ### Evals
    eval_tokens = activation_store.get_batch_tokens()
    
    # Get Reconstruction Score
    recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(sparse_autoencoder, model, activation_store, eval_tokens)
    
    # get cache
    _, cache = model.run_with_cache(eval_tokens, prepend_bos=False, names_filter=[get_act_name("pattern", hook_point_layer), hook_point])
    
    # get act
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        original_act = cache[sparse_autoencoder.cfg.hook_point][:,:,sparse_autoencoder.cfg.hook_point_head_index]
    else:
        original_act = cache[sparse_autoencoder.cfg.hook_point]
        
    transcoder_out, feature_acts, _, _, _, _ = sparse_autoencoder(
        original_act
    )
    patterns_original = cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
    del cache
    
    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()
    
    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(transcoder_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in
    
    wandb.log(
        {

            # l2 norms
            "metrics/l2_norm": l2_norm_out.mean().item(),
            "metrics/l2_ratio": l2_norm_ratio.mean().item(),
            
            # CE Loss
            "metrics/CE_loss_score": recons_score,
            "metrics/ce_loss_without_sae": ntp_loss,
            "metrics/ce_loss_with_sae": recons_loss,
            "metrics/ce_loss_with_ablation": zero_abl_loss,
            
        },
        step=n_training_steps,
    )
    
    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:,:,head_index])[0].to(activations.dtype)
        activations[:,:,head_index] = new_actions
        return activations

    head_index = sparse_autoencoder.cfg.hook_point_head_index
    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
    
    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
        _, new_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_reconstructed = new_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del new_cache
        
    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
        _, zero_ablation_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_ablation = zero_ablation_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del zero_ablation_cache
        
        
    # Visualizations to show L0 / MSE distributions
    # l0 = (feature_acts > 0).float().sum(-1)
    # per_token_l2_loss = (sae_out - original_act).pow(2).sum(dim=-1).squeeze()
    
    # fig = px.scatter(
    #     x = per_token_l2_loss.flatten().cpu().numpy(),
    #     y = l0.flatten().cpu().numpy(),
    #     color = np.arange(per_token_l2_loss.shape[1]).repeat(per_token_l2_loss.shape[0]),
    #     opacity=0.5,
    #     labels = {"color": "position", "x": "MSE Loss", "y": "L0"},
    #     title = "L0 vs MSE Loss",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    # )
    # wandb.log({"plots/l0_vs_mse_loss": wandb.Plotly(fig)}, step = n_training_steps)
    
    # fig = px.scatter(
    #     x =  per_token_l2_loss.flatten().cpu().numpy(),
    #     y = l2_norm_in.flatten().cpu().numpy(),
    #     color = np.arange(per_token_l2_loss.shape[1]).repeat(per_token_l2_loss.shape[0]),
    #     opacity=0.5,
    #     labels={"color": "position", "x": "MSE Loss", "y": "L2 Norm"},
    #     title = "L2 Norm vs MSE Loss",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    # )
    # wandb.log({"plots/l2_norm_vs_mse_loss": wandb.Plotly(fig)}, step = n_training_steps)

    # if dealing with a head SAE, do the head metrics.
    if sparse_autoencoder.cfg.hook_point_head_index:
        
        # show patterns before/after
        # fig_patterns_original = px.imshow(patterns_original[0].numpy(), title="original attn scores",
        #     color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_original.update_layout(coloraxis_showscale=False)         # hide colorbar 
        # wandb.log({"attention/patterns_original": wandb.Plotly(fig_patterns_original)}, step = n_training_steps)
        # fig_patterns_reconstructed = px.imshow(patterns_reconstructed[0].numpy(), title="reconstructed attn scores",
        #         color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_reconstructed.update_layout(coloraxis_showscale=False)         # hide colorbar
        # wandb.log({"attention/patterns_reconstructed": wandb.Plotly(fig_patterns_reconstructed)}, step = n_training_steps)
        
        kl_result_reconstructed = kl_divergence_attention(patterns_original, patterns_reconstructed)
        kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
        
        kl_result_ablation = kl_divergence_attention(patterns_original, patterns_ablation)
        kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        # #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
    
        wandb.log(
            {

              "metrics/kldiv_reconstructed": kl_result_reconstructed.mean().item(),
              "metrics/kldiv_ablation": kl_result_ablation.mean().item(),
                
            },
            step=n_training_steps,
        )

@torch.no_grad()
def get_recons_loss(sparse_autoencoder, model, activation_store, batch_tokens):
    hook_point = activation_store.cfg.hook_point
    loss = model(batch_tokens, return_type="loss")

    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:,:,head_index])[0].to(activations.dtype)
        activations[:,:,head_index] = new_actions
        return activations

    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1]).to(mlp_post.dtype)
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)