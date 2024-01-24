class SparseQK(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_hidden"]
        reg_coeff = cfg["reg_coeff"]
        t.manual_seed(cfg["seed"])
        n_heads = cfg["n_heads"]
        self.n_heads = n_heads
        self.d_head = cfg["d_head"]
        self.d_model = cfg["d_model"]

        self.d_hidden = d_hidden
        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg["d_model"], 2, self.d_hidden)))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(d_hidden, self.n_heads)))
        self.b_enc = nn.Parameter(t.zeros(2, self.d_hidden))
        self.b_dec = nn.Parameter(t.zeros(self.n_heads))
        self.eps = cfg["eps"]

        #self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        #self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        #self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.reg_coeff = reg_coeff
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device="cuda"))

        self.to(cfg["device"])

    def forward(self, x, masked = False):
        expanded_q = x.unsqueeze(2).expand(-1, -1, x.size(1), -1)
        expanded_k = x.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        # x = [batch, posn_q, posn_k, 2, d_model]
        x = t.stack([expanded_q, expanded_k], dim=3)
        acts = einops.einsum(x, self.W_enc, "batch posn_q posn_k qk d_model, d_model qk d_hidden -> batch posn_q posn_k qk d_hidden") + self.b_enc
        acts = F.relu(acts)
        acts = pairwise_product = acts[:, :, :, 0, :] * acts[:, :, :, 1, :]
        reg_loss = self.reg_coeff * t.sqrt(acts.float().abs() + self.eps).sum()
        feature_fires = (acts > 0).sum()
        score_reconstr = einops.einsum(acts, self.W_dec, "batch posn_q posn_k d_hidden, d_hidden n_heads -> batch posn_q posn_k n_heads")/(self.d_head ** 0.5) + self.b_dec
        score_reconstr = einops.rearrange(score_reconstr, "batch posn_q posn_k n_heads -> batch n_heads posn_q posn_k")
        if masked:
          score_recostr = self.apply_causal_mask(score_reconstr)
        return score_reconstr, reg_loss, feature_fires, never_fired

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores



    @t.no_grad()
    def renorm_weights(self):
        q_normed = self.W_enc[:, 0, :].norm(dim = 0)
        k_normed = self.W_enc[:, 1, :].norm(dim = 0)
        self.W_enc[:, 0, :] = self.W_enc[:, 0, :]/q_normed
        self.W_enc[:, 1, :] = self.W_enc[:, 1, :]/k_normed


def train_sparse_QK(
    orig_model,
    cfg,
    n_epochs: int,
    layer,
    data,
    dead_freq):
    
    
    sparse_model = SparseQK(cfg = cfg).cuda()
    print(f"Training model with {sparse_model.d_hidden} feature pairs.")
    optimizer = t.optim.AdamW(sparse_model.parameters(), lr = 1e-3)
    wandb.init(project="sparse_QK_gpt2_L0.5", entity="kwyn390")
    feature_tots = t.zeros(cfg.d_hidden)
    tot_data = 0
    for epoch in range(n_epochs):
        batch_index = None
        progress_bar = tqdm(list(enumerate(data)))
        for batch_idx, batch in progress_bar:
            if batch_idx < 30000:
              tot_data += (batch.size(0) * batch.size(1) * batch.size(1))
              #normalise encoder weights
              sparse_model.renorm_weights()
              optimizer.zero_grad()
              _, cache = orig_model.run_with_cache(batch["tokens"])
              resid_pre = cache["resid_pre", layer].clone()
              ln = cache['blocks.'+str(layer)+'.ln1.hook_scale']
              resid_pre = resid_pre/ln
              q, k = cache["q", 10], cache["k", 10]
              original_scores = einops.einsum(q, k, "batch pos_q n_heads d_head, batch pos_k n_heads d_head -> batch n_heads pos_q pos_k").clone()/8
              modified_output, reg_loss, feature_fires, dead = sparse_model(resid_pre)
              mse_loss = t.nn.MSELoss(reduction="mean")
              reconstruction_loss = mse_loss(modified_output, original_scores)
              loss = reconstruction_loss + reg_loss
              loss.backward(retain_graph = True)
              optimizer.step()
              feature_tots += feature_fires.sum(0).sum(0).sum(0)
              feature_freqs = feature_tots / tot_data
              wandb.log({
                  "recons_score": reconstruction_loss,
                  "loss": loss,
                  "reg_loss": reg_loss,
                  "l0": feature_fires / (feature_fires.shape(0)*(feature_fires.shape(1)**2)),
                  "dead_features": (feature_freqs < dead_freq).sum()

              })

        print(
                f"Epoch {epoch} reconstruction loss: {reconstruction_loss.item()} l0: {l0} reg_loss {reg_loss}"
            )
        print(f"Epoch {epoch} loss: {loss.item()}")

        del batch_index


    return sparse_model