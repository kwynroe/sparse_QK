

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
        is_query=False,
    ):
        super().__init__()
        self.is_query = is_query
        self.cfg = cfg
        self.layer = cfg.layer
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}")
        self.d_out = cfg.d_out
        if not isinstance(self.d_out, int):
            raise ValueError(f"d_out must be an int but was {self.d_out=}; {type(self.d_out)=}")
        self.d_hidden = cfg.d_hidden
        self.dtype = cfg.dtype
        self.device = cfg.device
        self.eps = cfg.eps
        self.d_head = cfg.d_head
        self.n_head = cfg.n_head

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
        if cfg.norming_decoder_during_training:
            self.set_decoder_norm_to_unit_norm()
        self.b_pre = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        self.b_out = nn.Parameter(torch.zeros(self.d_out, dtype=self.dtype, device=self.device))

        self.hook_transcoder_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_transcoder_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, dead_neuron_mask=None):
        # move x to correct dtype
        x = x.to(self.dtype)
        transcoder_in = self.hook_transcoder_in(
            x - self.b_pre * (not self.cfg.disable_b_pre)
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

        transcoder_out = self.hook_transcoder_out(einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_hidden, d_hidden d_in -> ... d_in",
            ) + self.b_out
        )
        
        lp_norm = self.reg_loss(feature_acts)

        return transcoder_out, feature_acts, lp_norm
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()
    

    def reg_loss(self, feature_acts, p=0.5):
        "Scaled regularisation term."
        if self.cfg.norming_decoder_during_training:
            scaled_feature_acts = feature_acts
        else:
            scaled_feature_acts = feature_acts * torch.norm(self.W_dec, dim=-1)
        return torch.sqrt(scaled_feature_acts.float().abs() + self.eps).sum()    # this isnt actually the l0.5 norm, but its sqrt??

    @torch.no_grad()
    def initialize_biases(self, activation_store, model_W=None, model_b=None):
        if self.cfg.biases_init_method == "mean":
            if model_W is None or model_b is None:
                raise ValueError("Model weights required for mean initialisation in initialise_b_pre.")

            previous_b_pre = self.b_pre.clone().cpu()
            previous_b_out = self.b_out.clone().cpu()
            b_pre = initialize_b_pre_with_mean(previous_b_pre, activation_store)
            b_out = initialize_b_out_with_mean(previous_b_out, activation_store, model_W, model_b)
            self.b_pre.data = b_pre.to(self.dtype).to(self.device)
            self.b_out.data = b_out.to(self.dtype).to(self.device)
        elif self.cfg.biases_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unexpected biases_init_method: {self.cfg.biases_init_method}")
        

    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model, replace):
        """
        A method for running the model with the Transcoder activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        input_hook = self.cfg.hook_transcoder_in_q if self.is_query else self.cfg.hook_transcoder_in_k
        target_hook = self.cfg.hook_transcoder_out_q if self.is_query else self.cfg.hook_transcoder_out_k

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
    def load_from_pretrained(cls, path: str, is_query=True):
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
        instance = cls(cfg=state_dict["cfg"], is_query=is_query)
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        this_type = self.cfg.type_q if self.is_query else self.cfg.type_k
        transcoder_name = (
            f"sparse_transcoder_{self.cfg.model_name}_{this_type}_{self.cfg.d_hidden}"
        )
        return transcoder_name




@torch.no_grad()
def initialize_b_pre_with_mean(previous_b_pre, activation_store):

    all_activations = activation_store.storage_buffer.detach().cpu()
    out = all_activations.mean(dim=0)

    previous_distances = torch.norm(all_activations - previous_b_pre, dim=-1)
    distances = torch.norm(all_activations - out, dim=-1)

    print("Reinitializing b_pre with mean of activations")
    print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
    print(f"New distances: {distances.median(0).values.mean().item()}")
    
    return out


@torch.no_grad()
def initialize_b_out_with_mean(previous_b_out, activation_store, model_W, model_b):

    all_activations = activation_store.storage_buffer.detach()
    all_activations = einops.einsum(all_activations, model_W, "... d_model, n_head d_model d_head -> ... n_head d_head") + model_b
    all_activations = einops.rearrange(all_activations, "... n_head d_head -> ... (n_head d_head)")
    all_activations = all_activations.cpu()
    out = all_activations.mean(dim=0)

    previous_distances = torch.norm(all_activations - previous_b_out, dim=-1)
    distances = torch.norm(all_activations - out, dim=-1)

    print("Reinitializing b_out with mean of activations")
    print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
    print(f"New distances: {distances.median(0).values.mean().item()}")

    return out