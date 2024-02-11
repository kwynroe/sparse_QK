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
    sparse_transcoder1 = SparseTranscoder(cfg1)
    sparse_transcoder2 = SparseTranscoder(cfg2)
    activations_loader = ActivationsStore(cfg1, model)

    if cfg1.log_to_wandb:
        wandb.init(project=cfg1.wandb_project, config=cfg1, name=cfg1.run_name)

    # train SAE
    sparse_transcoder1, sparse_transcoder2 = train_transcoder_on_language_model_parallel(
        cfg1,
        model,
        sparse_transcoder1,
        sparse_transcoder2,
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
    path = f"{cfg1.checkpoint_path}/final_{sparse_transcoder1.get_name()}.pt"
    sparse_transcoder1.save_model(path)
    path = f"{cfg2.checkpoint_path}/final_{sparse_transcoder2.get_name()}.pt"
    sparse_transcoder2.save_model(path)
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

    return sparse_transcoder1, sparse_transcoder2
