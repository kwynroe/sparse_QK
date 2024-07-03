import os

import torch
import transformer_lens
import wandb

from ActivationStoreParallel import ActivationsStore
from sparse_transcoder import SparseTranscoder

# from sae_training.activation_store import ActivationStore
from transcoder_training_parallel import train_transcoder_on_language_model_parallel


def language_model_transcoder_runner_parallel(cfg1, cfg2):
    """ """

    model = transformer_lens.HookedTransformer.from_pretrained(cfg1.model_name, fold_ln=True)
    W_Q, b_Q = model.W_Q[cfg1.layer], model.b_Q[cfg1.layer]
    W_K, b_K = model.W_K[cfg1.layer], model.b_K[cfg1.layer]
    query_transcoder = SparseTranscoder(cfg1, W_Q, b_Q)
    key_transcoder = SparseTranscoder(cfg2, W_K, b_K)
    activations_loader = ActivationsStore(cfg1, model)

    if cfg1.log_to_wandb:
        wandb.init(entity = cfg1.entity, project=cfg1.wandb_project, config=cfg1, name=cfg1.run_name)

    # train SAE
    query_transcoder, key_transcoder = train_transcoder_on_language_model_parallel(
        cfg1,
        model,
        query_transcoder,
        key_transcoder,
        activations_loader,
        n_checkpoints=cfg1.n_checkpoints,
        batch_size=cfg1.train_batch_size,
        feature_sampling_method=cfg1.feature_sampling_method,
        feature_sampling_window=cfg1.feature_sampling_window,
        feature_reinit_scale=cfg1.feature_reinit_scale,
        dead_feature_threshold=cfg1.dead_feature_threshold,
        dead_feature_window=cfg1.dead_feature_window,
        use_wandb=cfg1.log_to_wandb,
        wandb_log_frequency=cfg1.wandb_log_frequency,
    )

    # save sae to checkpoints folder
    path = f"{cfg1.checkpoint_path}/final_{query_transcoder.get_name()}.pt"
    query_transcoder.save_model(path)
    path = f"{cfg2.checkpoint_path}/final_{key_transcoder.get_name()}.pt"
    key_transcoder.save_model(path)
    # upload to wandb
    """
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sparse_transcoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])
        """

    if cfg1.log_to_wandb:
        wandb.finish()

    return query_transcoder, key_transcoder
