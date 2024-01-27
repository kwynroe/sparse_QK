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
        d_hidden = cfg["dict_size"]
        reg_coeff = cfg["reg_coeff"]
        t.manual_seed(cfg["seed"])
        self.cfg = cfg
        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg["act_size"], d_hidden)))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(d_hidden, cfg["act_size"])))
        self.b_dec = nn.Parameter(t.zeros(cfg["act_size"]))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.d_hidden = d_hidden
        self.l1_coeff = reg_coeff

        self.to(cfg["device"])

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        reg_loss = self.reg_coeff * t.sqrt(acts.float().abs() + self.eps).sum()
        feature_fires = (acts > 0).int()
        loss = l2_loss + reg_loss
        return loss, x_reconstruct, l2_loss, reg_loss, feature_fires

    @t.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
        
        
def train_sparse_query_transcoder(
    orig_model,
    cfg,
    n_epochs: int,
    layer,
    data,
    scaled = 'none'):
      
    sparse_model = Transcoder(cfg = cfg).cuda()
    print(f"Training model with {sparse_model.d_hidden} feature pairs.")
    optimizer = t.optim.AdamW(sparse_model.parameters(), lr = 1e-3)
    wandb.init(project="sparse_query_transcoder", entity="kwyn390")
    feature_tots = t.zeros(cfg["d_hidden"]).cuda()
    tot_data = 0
    for epoch in range(n_epochs):
        batch_index = None
        progress_bar = tqdm(list(enumerate(data)))
        for batch_idx, batch in progress_bar:
              tot_data += (batch["tokens"].size(0) * batch["tokens"].size(1) * batch["tokens"].size(1))
              #normalise encoder weights
              sparse_model.make_decoder_weights_and_grad_unit_norm()
              optimizer.zero_grad()
              _, cache = orig_model.run_with_cache(batch["tokens"])
              resid_pre = cache["resid_pre", layer].clone()
              ln = cache['blocks.'+str(layer)+'.ln1.hook_scale']
              resid_pre = resid_pre/ln
              loss, modified_output, l2_loss, reg_loss, feature_fires = sparse_model(resid_pre)
              original_queries = cache["q", layer].flatten(-2)
              mse_loss = t.nn.MSELoss(reduction="mean")
              reconstruction_loss = mse_loss(original_queries, modified_output)
              loss = reconstruction_loss + reg_loss
              loss.backward(retain_graph = True)
              optimizer.step()
              feature_tots += feature_fires.sum(0).sum(0).sum(0)
              l0 = feature_fires.sum() / (feature_fires.size(0)*(feature_fires.size(1)**2))
              feature_freqs = feature_tots / tot_data
              wandb.log({
                  "recons_score": reconstruction_loss,
                  "loss": loss,
                  "reg_loss": reg_loss,
                  "l0": l0,
                  "dead_features": (feature_freqs < cfg["dead_freq"]).sum()

              })
              del batch_idx
        print(
                f"Epoch {epoch} reconstruction loss: {reconstruction_loss.item()} l0: {l0} reg_loss {reg_loss}"
            )
        print(f"Epoch {epoch} loss: {loss.item()}")

    return sparse_model
