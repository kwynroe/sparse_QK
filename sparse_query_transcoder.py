import torch as t
from torch import Tensor, nn
from torch.nn.functional import normalize
import transformer_lens
from transformer_lens import HookedTransformer
from dataclasses import dataclass
import math
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import einops

class Transcoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_hidden"]
        reg_coeff = cfg["reg_coeff"]
        t.manual_seed(cfg["seed"])
        self.cfg = cfg
        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg["d_model"], d_hidden)))
        self.b_enc = nn.Parameter(t.zeros(cfg["d_hidden"]))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(d_hidden, cfg["d_model"])))
        self.b_dec = nn.Parameter(t.zeros(cfg["d_model"]))
        self.d_hidden = d_hidden
        self.reg_coeff = reg_coeff
        self.eps = cfg["eps"]
        self.to(cfg["device"])

    def forward(self, x, target, dead_neuron_mask):
        x_cent = x - self.b_dec
        acts = (x_cent @ self.W_enc + self.b_enc)
        relu_acts = F.relu(acts)
        sae_out = relu_acts @ self.W_dec + self.b_dec
        l2_loss = (sae_out.float() - target.float()).pow(2).sum(-1).mean(0)
        reg_loss = self.reg_coeff * t.sqrt(acts.float().abs() + self.eps).sum()
        feature_fires = (acts > 0).int()
        mse_loss_ghost_resid = t.tensor(0.0).cuda()
        # gate on config and training so evals is not slowed down.
        if self.cfg["use_ghost_grads"] and self["training"] and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None 
            
            # ghost protocol
            
            # 1.
            residual = target - sae_out
            l2_norm_residual = t.norm(residual, dim=-1)
            
            # 2.
            feature_acts_dead_neurons_only = t.exp(acts[:, dead_neuron_mask])
            ghost_out =  feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask,:]
            l2_norm_ghost_out = t.norm(ghost_out, dim = -1)
            norm_scaling_factor = l2_norm_residual / (l2_norm_ghost_out* 2)
            ghost_out = ghost_out*norm_scaling_factor[:, None].detach()
            
            # 3. 
            mse_loss_ghost_resid = (
                t.pow((ghost_out - residual.float()), 2) / (residual**2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (l2_loss / mse_loss_ghost_resid).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        loss = l2_loss + reg_loss + mse_loss_ghost_resid.mean()
        return sae_out, loss, l2_loss, reg_loss, mse_loss_ghost_resid.mean(), feature_fires

    @t.no_grad()
    def renorm_weights(self):
        self.W_dec.data /= t.norm(self.W_dec.data, dim=1, keepdim=True)  
        
def train_sparse_transcoder(
    orig_model,
    cfg,
    n_epochs: int,
    layer,
    data,
    comp,
    scaled = 'none'):
      
    sparse_model = Transcoder(cfg = cfg).cuda()
    print(f"Training model with {sparse_model.d_hidden} feature pairs.")
    optimizer = t.optim.AdamW(sparse_model.parameters(), lr = 1e-3)
    wandb.init(project="sparse_query_transcoder", entity="kwyn390")
    feature_tots = t.zeros(cfg["d_hidden"]).cuda()
    tot_data = 0
    dead_feature_indices = t.zeros(cfg["d_hidden"])
    for epoch in range(n_epochs):
        batch_index = None
        progress_bar = tqdm(list(enumerate(data)))
        for batch_idx, batch in progress_bar:
              tot_data += (batch["tokens"].size(0) * batch["tokens"].size(1) * batch["tokens"].size(1))
              #normalise encoder weights
              sparse_model.renorm_weights()
              optimizer.zero_grad()
              _, cache = orig_model.run_with_cache(batch["tokens"])
              resid_pre = cache["resid_pre", layer].clone()
              ln = cache['blocks.'+str(layer)+'.ln1.hook_scale']
              resid_pre = resid_pre/ln
              original_comp = cache[comp, layer].flatten(-2)
              _, l2_loss, reg_loss, feature_fires = sparse_model(resid_pre, original_comp, dead_feature_indices)
              loss = l2_loss + reg_loss
              loss.backward(retain_graph = True)
              optimizer.step()
              feature_tots += feature_fires.sum(0).sum(0).sum(0)
              l0 = feature_fires.sum() / (feature_fires.size(0)*(feature_fires.size(1)))
              feature_freqs = feature_tots / tot_data
              dead_feature_indices = (feature_freqs < cfg["dead_freq"])
              wandb.log({
                  "l2": l2_loss,
                  "loss": loss,
                  "reg_loss": reg_loss,
                  "l0": l0,
                  "dead_features": (feature_freqs < cfg["dead_freq"]).sum()

              })
              del batch_idx
        print(
                f"Epoch {epoch} reconstruction loss: {l2_loss.item()} l0: {l0} reg_loss {reg_loss}"
            )
        print(f"Epoch {epoch} loss: {loss.item()}")

    return sparse_model
