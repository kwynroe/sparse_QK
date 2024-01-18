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

    
    @t.no_grad()
    def resample_neurons_anthropic(
        self, 
        dead_neuron_indices, 
        model,
        optimizer, 
        activation_store):
        """
        Arthur's version of Anthropic's feature resampling
        procedure.
        """
        # collect global loss increases, and input activations
        global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
            model, activation_store
        )

        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = t.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        # if we don't have enough samples for for all the dead neurons, take the first n
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[:sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / t.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[dead_neuron_indices, :].T
        self.b_enc.data[dead_neuron_indices] = 0.0
        
        # Reset the Encoder Weights
        if dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(dead_neuron_indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale * average_norm

            # Set biases to resampled value
            relevant_biases = self.b_enc.data[dead_neuron_indices].mean()
            self.b_enc.data[dead_neuron_indices] = relevant_biases * 0 # bias resample factor (put in config?)

        else:
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale
            self.b_enc.data[dead_neuron_indices] = -5.0
        
        # TODO: Refactor this resetting to be outside of resampling.
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, dead_neuron_indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][dead_neuron_indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_in)
                    v[v_key][dead_neuron_indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_in,)
                else:
                    raise ValueError(f"Unexpected dict_idx {dict_idx}")
                
    @t.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store):
        """
        Collects the losses for resampling neurons (anthropic)
        """
        
        batch_size = self.cfg.store_batch_size
        
        # we're going to collect this many forward passes
        number_final_activations = self.cfg.resample_batches * batch_size
        # but have seq len number of tokens in each
        number_activations_total = number_final_activations * self.cfg.context_size
        anthropic_iterator = range(0, number_final_activations, batch_size)
        anthropic_iterator = tqdm(anthropic_iterator, desc="Collecting losses for resampling...")
        
        global_loss_increases = torch.zeros((number_final_activations,), dtype=self.dtype, device=self.device)
        global_input_activations = torch.zeros((number_final_activations, self.d_in), dtype=self.dtype, device=self.device)

        for refill_idx in anthropic_iterator:
            
            # get a batch, calculate loss with/without using SAE reconstruction.
            batch_tokens = activation_store.get_batch_tokens()
            ce_loss_with_recons = self.get_test_loss(batch_tokens, model)
            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type = "loss",
                loss_per_token = True,
            )
            # ce_loss_without_recons = model.loss_fn(normal_logits, batch_tokens, True)
            # del normal_logits
            
            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[:,:,self.cfg.hook_point_head_index]

            # calculate the difference in loss
            changes_in_loss = ce_loss_with_recons - ce_loss_without_recons
            changes_in_loss = changes_in_loss.cpu()
            
            # sample from the loss differences
            probs = F.relu(changes_in_loss) / F.relu(changes_in_loss).sum(dim=1, keepdim=True)
            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()
            
            assert samples.shape == (batch_size,), f"{samples.shape=}; {self.cfg.store_batch_size=}"
            
            end_idx = refill_idx + batch_size
            global_loss_increases[refill_idx:end_idx] = changes_in_loss[torch.arange(batch_size), samples]
            global_input_activations[refill_idx:end_idx] = normal_activations[t.arange(batch_size), samples]
        
        return global_loss_increases, global_input_activations
           
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, dead_neuron_indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return 



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