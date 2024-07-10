import os

import torch
import transformer_lens
import wandb

from ActivationStoreParallel import ActivationsStore
from sparse_transcoder import SparseTranscoder

# from sae_training.activation_store import ActivationStore
from transcoder_training_parallel import train_transcoder_on_language_model_parallel


def language_model_transcoder_runner_parallel(cfg):
    "Wrapper around transcoder training."
    # Initialise model.
    model = transformer_lens.HookedTransformer.from_pretrained(cfg.model_name, fold_ln=True)
    activations_loader = ActivationsStore(cfg, model)

    # Create transcoders.
    query_transcoder = SparseTranscoder(cfg, is_query=True)
    key_transcoder = SparseTranscoder(cfg, is_query=False)

    if cfg.log_to_wandb:
        wandb.init(entity=cfg.entity, project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # train SAE.
    query_transcoder, key_transcoder = train_transcoder_on_language_model_parallel(
        cfg,
        model,
        query_transcoder,
        key_transcoder,
        activations_loader
    )

    # save transcoder.
    path_q = f"{cfg.checkpoint_path}/final_{query_transcoder.get_name()}.pt"
    query_transcoder.save_model(path_q)
    path_k = f"{cfg.checkpoint_path}/final_{key_transcoder.get_name()}.pt"
    key_transcoder.save_model(path_k)


    if cfg.log_to_wandb and cfg.log_final_model_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sparse_transcoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    if cfg.log_to_wandb:
        wandb.finish()

    return query_transcoder, key_transcoder