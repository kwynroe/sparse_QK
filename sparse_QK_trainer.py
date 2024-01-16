class SparseQK(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_hidden"]
        l1_coeff = cfg["l1_coeff"]
        t.manual_seed(cfg["seed"])
        n_heads = cfg["n_heads"]
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(cfg["d_model"], 2, self.d_hidden)))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(d_hidden, self.n_heads)))
        self.b_enc = nn.Parameter(t.zeros(2, self.d_hidden))
  
        #self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        #self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        #self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])

    def forward(self, x):
        expanded_q = x.unsqueeze(2).expand(-1, -1, x.size(1), -1)
        expanded_k = x.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        # x = [batch, posn_q, posn_k, 2, d_model]
        x = t.stack([expanded_q, expanded_k], dim=3)
        acts = einops.einsum(x, self.W_enc, "batch posn_q posn_k qk d_model, d_model qk d_hidden -> batch posn_q posn_k qk d_hidden") + self.b_enc
        acts = pairwise_product = acts[:, :, :, 0, :] * acts[:, :, :, 1, :]
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        l0_norm = (acts > 0).sum() / (acts.shape[0]*acts.shape[1]**2)
        patt_reconstr = einops.einsum(acts, self.W_dec, "batch posn_q posn_k d_hidden, d_hidden n_heads -> batch n_heads posn_q posn_k")
        return t.tril(patt_reconstr), l1_loss, l0_norm


def train_sparse_QK(
    orig_model,
    n_epochs,
    layer,
    data,
    cfg):

    train_loss_list = []
    indices = []
    sparse_model = SparseQK(cfg = cfg).cuda()
    print(f"Training model with {sparse_model.d_hidden} feature pairs.")
    optimizer = t.optim.AdamW(sparse_model.parameters(), lr = 1e-3)
    wandb.init(project="sparse_QK_gpt2", entity="kwyn390")
    for epoch in range(n_epochs):
        batch_index = None
        progress_bar = tqdm(list(enumerate(data)))
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()
            _, cache = orig_model.run_with_cache(batch["tokens"])
            normalised_resid_pre = cache["resid_pre", layer].clone()
            original_scores = t.tril(cache["attn_scores", layer].clone())
            modified_output, l1, l0 = sparse_model(normalised_resid_pre)

            mse_loss = t.nn.MSELoss(reduction="mean")
            reconstruction_loss = mse_loss(modified_output, original_scores)
            loss = reconstruction_loss + l1
            loss.backward(retain_graph = True)
            optimizer.step()
            wandb.log({
                "recons_score": reconstruction_loss,
                "loss": loss,
                "l1": l1,
                "l0": l0,

            })
        print(
                f"Epoch {epoch} reconstruction loss: {reconstruction_loss.item()} l0: {l0} l1 {l1}"
            )
        print(f"Epoch {epoch} loss: {loss.item()}")
        del batch_index
        indices.append(epoch)
        train_loss_list.append(loss.item())

    return sparse_model