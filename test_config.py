from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class UnifiedConfig():
    # Common settings
    model_name: str = "tiny-stories-1M"
    hook_point: str = "blocks.4.hook_resid_pre"
    ln: str = 'blocks.4.ln1.hook_scale'
    hook_point_layer: int = 4
    layer: int = 4
    d_in: int = 64
    d_out: int = 64
    n_head: int = 16
    d_head: int = 8
    dataset_path: str = "Skylion007/openwebtext"
    is_dataset_tokenized: bool = False
    training: bool = True
    attn_scores_normed = True
    
    # SAE Parameters
    expansion_factor: int = 12   # TODO: NOT being used??
    d_hidden: int = 127
    b_dec_init_method: str = "mean"
    norming_decoder_during_training = False
    
    # Training Parameters
    lr: float = 1e-5
    reg_coefficient: float = 4e-6
    lr_scheduler_name: Optional[str] = None
    train_batch_size: int = 1024
    context_size: int = 256
    lr_warm_up_steps: int = 100
    
    # Activation Store Parameters
    n_batches_in_buffer: int = 128
    total_training_tokens: int = 2_000
    store_batch_size: int = 32
    use_cached_activations: bool = False
    
    # Resampling protocol
    feature_sampling_method: str = 'none'
    feature_sampling_window: int = 1000
    feature_reinit_scale: float = 0.2
    resample_batches: int = 1028
    dead_feature_window: int = 50000
    dead_feature_threshold: float = 1e-6
    
    # WANDB
    log_to_wandb: bool = False
    log_final_model_to_wandb: bool = False
    wandb_project: str = "sparsification"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 50
    entity: str = "biggs-University College London (UCL)"
    
    # Misc
    device: str = "cuda"
    eps: float = 1e-7
    seed: int = 42
    reshape_from_heads: bool = True
    n_checkpoints: int = 10
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = torch.float32
    run_name: str = "qk_parallel"
    
    # Query-specific settings
    hook_transcoder_in_q: str = "blocks.4.hook_resid_pre"
    hook_transcoder_out_q: str = "blocks.4.attn.hook_q"
    target_q: str = "blocks.4.attn.hook_q"
    type_q: str = "resid_to_queries"
    
    # Key-specific settings
    hook_transcoder_in_k: str = "blocks.4.hook_resid_pre"
    hook_transcoder_out_k: str = "blocks.4.attn.hook_k"
    target_k: str = "blocks.4.attn.hook_k"
    type_k: str = "resid_to_keys"

test_cfg = UnifiedConfig()
test_cfg.run_name = f"{test_cfg.d_hidden}_{test_cfg.reg_coefficient}_{test_cfg.lr}"
    
# Ensure TestConfig is exported
__all__ = ['TestConfig']