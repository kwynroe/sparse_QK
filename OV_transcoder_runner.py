import os

import torch
import transformer_lens
import wandb

from ActivationStoreParallel import ActivationsStore
from sparse_transcoder_OV import SparseTranscoder_OV

# from sae_training.activation_store import ActivationStore
from OV_transcoder_train import train_OV_transcoder


def OV_transcoder_runner_parallel(cfg, key_transcoder):
    "Wrapper around transcoder training."
    print("Running...")

    # Load model.
    model = transformer_lens.HookedTransformer.from_pretrained(cfg.model_name, fold_ln=True)
    activations_store = ActivationsStore(cfg, model)

    # Create and initialise transcoders.
    transcoder = SparseTranscoder_OV(cfg)

    if cfg.log_to_wandb:
        wandb.init(entity=cfg.entity, project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # train SAE.
    ov_transcoder = train_OV_transcoder(
        cfg,
        model,
        transcoder,
        key_transcoder,
        activations_store,
        cfg.train_batch_size
        )

    # save transcoder.
    path = f"{cfg.checkpoint_path}/final_{ov_transcoder.get_name()}.pt"
    ov_transcoder.save_model(path)


    if cfg.log_to_wandb:
        wandb.finish()

    return ov_transcoder