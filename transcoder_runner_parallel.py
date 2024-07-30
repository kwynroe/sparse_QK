import wandb
from transformer_lens import HookedTransformer
from ActivationStoreParallel import ActivationsStore
from sparse_transcoder import SparseTranscoder
from transcoder_training_parallel import train_transcoder_on_language_model_parallel


def language_model_transcoder_runner_parallel(cfg, model: HookedTransformer, activations_store: ActivationsStore):
    """Wrapper around transcoder training."""
    print("Running transcoder training...")

    # Create and initialize transcoders
    query_transcoder = SparseTranscoder(cfg, is_query=True)
    key_transcoder = SparseTranscoder(cfg, is_query=False)
    query_transcoder.initialize_biases(activations_store, model.W_Q[cfg.layer], model.b_Q[cfg.layer])
    key_transcoder.initialize_biases(activations_store, model.W_K[cfg.layer], model.b_K[cfg.layer])

    cfg.attn_scores_norm = cfg.d_head ** 0.5 if cfg.attn_scores_normed else 1

    if cfg.log_to_wandb:
        wandb.init(entity=cfg.entity, project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # Train SAE
    query_transcoder, key_transcoder = train_transcoder_on_language_model_parallel(
        cfg,
        model,
        query_transcoder,
        key_transcoder,
        activations_store,
    )
    
    # Post-processing
    query_transcoder.fold_W_dec_norm()
    key_transcoder.fold_W_dec_norm()

    # Save transcoders
    _save_transcoders(cfg, query_transcoder, key_transcoder)

    if cfg.log_to_wandb:
        wandb.finish()

    return query_transcoder, key_transcoder

def _save_transcoders(cfg, query_transcoder, key_transcoder):
    """Helper function to save transcoders."""
    path_q = f"{cfg.checkpoint_path}/final_{query_transcoder.get_name()}.pt"
    path_k = f"{cfg.checkpoint_path}/final_{key_transcoder.get_name()}.pt"
    query_transcoder.save_model(path_q)
    key_transcoder.save_model(path_k)
    print(f"Saved query transcoder to {path_q}")
    print(f"Saved key transcoder to {path_k}")