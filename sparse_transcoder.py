

import gzip
import os
import pickle
from functools import partial

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint




class SparseTranscoder(HookedRootModule):
    """ """

    def __init__(
        self,
        cfg,
        W,
        b
    ):
        super().__init__()
        self.cfg = cfg
        self.layer = cfg.layer
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}")
        self.d_out = cfg.d_out
        if not isinstance(self.d_in, int):
            raise ValueError(f"d_out must be an int but was {self.d_out=}; {type(self.d_out)=}")
        self.d_hidden = cfg.d_hidden
        self.reg_coefficient = cfg.reg_coefficient
        self.dtype = cfg.dtype
        self.device = cfg.device
        self.eps = cfg.eps
        self.d_head = cfg.d_head
        self.n_head = cfg.n_head
        self.W = W
        self.b = b

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_hidden, dtype=self.dtype, device=self.device)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=self.dtype, device=self.device))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_hidden, self.d_out, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        self.b_dec_out = nn.Parameter(torch.zeros(self.d_out, dtype=self.dtype, device=self.device))

        self.hook_transcoder_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_transcoder_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, target, dead_neuron_mask=None):
        # move x to correct dtype
        x = x.to(self.dtype)
        transcoder_in = self.hook_transcoder_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                transcoder_in,
                self.W_enc,
                "... d_in, d_in d_hidden -> ... d_hidden",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        transcoder_out = einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_hidden, d_hidden d_in -> ... d_in",
            ) + self.b_dec_out
        

        # add config for whether l2 is normalized:
        mse_loss = (transcoder_out - target.float()).pow(2).sum(-1).mean()
  
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None

            # ghost protocol

            # 1.
            residual = target - transcoder_out
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (l2_norm_ghost_out * 2 + self.eps)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + self.eps)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
            
            mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        reg_loss = self.reg_coefficient * torch.sqrt(feature_acts.float().abs() + self.eps).sum()
        loss = mse_loss + reg_loss + mse_loss_ghost_resid

        return transcoder_out, feature_acts, loss, mse_loss, reg_loss, mse_loss_ghost_resid

    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        if self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(activation_store)
            self.initialize_b_dec_out_with_mean(activation_store)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}")


    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store):

        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activation_store.storage_buffer.detach().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def initialize_b_dec_out_with_mean(self, activation_store):

        previous_b_dec_out = self.b_dec_out.clone().cpu()
        all_activations = activation_store.storage_buffer.detach()
        all_activations = einops.einsum(all_activations, self.W, "... d_model, n_head d_model d_head -> ... n_head d_head") + self.b
        all_activations = einops.rearrange(all_activations, "... n_head d_head -> ... (n_head d_head)")
        all_activations = all_activations.cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec_out, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec_out with mean of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model, replace):
        """
        A method for running the model with the Transcoder activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        input_hook = self.cfg.hook_transcoder_in
        target_hook = self.cfg.hook_transcoder_out

        comp_cache = None

        def get_input_hook(activations, hook):
            global comp_cache
            comp_cache = activations
            return activations

        def replace_target_hook(activations, hook):
            global comp_cache
            new_output = self.forward(comp_cache.to(activations.dtype), activations)

        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(input_hook, get_input_hook)] + [(target_hook, replace_target_hook)],
        )

        return ce_loss_with_recons

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_hidden, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_hidden d_in, d_hidden d_in -> d_hidden",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_hidden, d_hidden d_in -> d_hidden d_in",
        )

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    @classmethod
    def load_from_pretrained(cls, path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps")
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl.gz file: {e}")
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError("The loaded state dictionary must contain 'cfg' and 'state_dict' keys")

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        transcoder_name = (
            f"sparse_transcoder_{self.cfg.model_name}_{self.cfg.type}_{self.cfg.d_hidden}"
        )
        return transcoder_name
