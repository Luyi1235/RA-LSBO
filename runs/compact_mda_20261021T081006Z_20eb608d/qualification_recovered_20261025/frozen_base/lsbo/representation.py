from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class StructuredVAE(nn.Module):
    def __init__(self, space, latent_dim=4, width=48):
        super().__init__()
        self.space, self.latent_dim = space, latent_dim
        self.embeddings = nn.ModuleDict({v.name: nn.Embedding(len(v.values), 4)
                                         for v in space.discrete if v.kind == "nominal"})
        input_dim = sum(3 if v.kind == "ordinal" else 5 for v in space.discrete)
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        self.decoder = nn.ModuleList([nn.Linear(latent_dim, len(v.values)) for v in space.discrete])
        self.performance = nn.Sequential(nn.Linear(2 * len(space.continuous) + latent_dim, width), nn.SiLU(),
                                         nn.Linear(width, 1))

    def discrete_mask(self, indices):
        masks = []
        positions = {v.name: i for i, v in enumerate(self.space.discrete)}
        for v in self.space.discrete:
            mask = torch.ones(indices.shape[0], dtype=torch.double)
            for parent, allowed in v.active_when.items():
                j = positions[parent]
                permitted = [self.space.discrete[j].values.index(value) for value in allowed]
                mask = mask * sum((indices[:, j] == i).double() for i in permitted)
            masks.append(mask)
        return torch.stack(masks, -1)

    def encode(self, indices):
        masks = self.discrete_mask(indices)
        parts = []
        for i, v in enumerate(self.space.discrete):
            mask = masks[:, i:i + 1]
            if v.kind == "ordinal":
                unit = indices[:, i:i + 1].double() / (len(v.values) - 1)
                embedding = torch.cat([2 * unit - 1, torch.sin(torch.pi * unit)], -1)
            else:
                embedding = self.embeddings[v.name](indices[:, i])
            parts.extend([embedding * mask, mask])
        hidden = self.encoder(torch.cat(parts, -1))
        return self.mu(hidden), self.logvar(hidden).clamp(-8, 4)

    def reconstruct_loss(self, indices, generator):
        mu, logvar = self.encode(indices)
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, generator=generator, dtype=torch.double)
        masks = self.discrete_mask(indices)
        losses = [F.cross_entropy(layer(z), indices[:, i], reduction="none") * masks[:, i]
                  for i, layer in enumerate(self.decoder)]
        reconstruction = torch.stack(losses, -1).sum() / masks.sum().clamp_min(1)
        reference = torch.randn(z.shape, generator=generator, dtype=torch.double)
        def kernel(a, b):
            distance = torch.cdist(a, b).square() / self.latent_dim
            return sum(torch.exp(-distance / scale) for scale in (0.5, 1.0, 2.0)) / 3
        mmd = kernel(z, z).mean() + kernel(reference, reference).mean() - 2 * kernel(z, reference).mean()
        return reconstruction + 0.05 * mmd


class Representation:
    def __init__(self, space, seed=0, latent_dim=4, width=48):
        self.space = space
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = StructuredVAE(space, latent_dim, width).double()
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.version = 0
        self.y_mean, self.y_std = 0.0, 1.0
        self.performance_fitted = False
        self.grammar = torch.tensor([space.indices(d) for d in space.configurations()], dtype=torch.long)

    def _indices(self, candidates):
        return torch.tensor([self.space.indices(c.discrete) for c in candidates], dtype=torch.long)

    def base_features(self, candidates):
        masks = np.vstack([self.space.mask(c.discrete) for c in candidates])
        units = np.vstack([self.space.unit(c) for c in candidates])
        return torch.tensor(np.hstack([units * masks, masks]), dtype=torch.double)

    def features(self, candidates):
        with torch.no_grad():
            mu, _ = self.model.encode(self._indices(candidates))
        return torch.cat([self.base_features(candidates), mu], -1)

    def latent(self, discrete):
        with torch.no_grad():
            mu, _ = self.model.encode(torch.tensor([self.space.indices(discrete)], dtype=torch.long))
        return mu[0]

    def fit(self, observations, epochs=40):
        usable = [o for o in observations if o.result.has_objective()]
        perf_x = perf_indices = target = None
        if len(usable) >= 3:
            candidates = [o.candidate for o in usable]
            energy = np.array([o.result.objective for o in usable])
            self.y_mean, self.y_std = float(energy.mean()), float(max(energy.std(), 1e-8))
            target = torch.tensor((energy - self.y_mean) / self.y_std, dtype=torch.double)
            perf_x, perf_indices = self.base_features(candidates), self._indices(candidates)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001 if self.version == 0 else 0.0004)
        for _ in range(epochs):
            order = torch.randperm(len(self.grammar), generator=self.generator)
            for batch in order.split(128):
                optimizer.zero_grad()
                loss = self.model.reconstruct_loss(self.grammar[batch], self.generator)
                if target is not None:
                    chosen = torch.randperm(len(usable), generator=self.generator)[:128]
                    latent, _ = self.model.encode(perf_indices[chosen])
                    prediction = self.model.performance(torch.cat([perf_x[chosen], latent], -1)).squeeze(-1)
                    loss = loss + F.smooth_l1_loss(prediction, target[chosen])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite representation loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
        self.model.eval()
        self.performance_fitted = target is not None
        self.version += 1

    def decode(self, z, rng, temperature=1.0, stochastic=False):
        with torch.no_grad():
            logits = [layer(torch.as_tensor(z, dtype=torch.double)).reshape(-1) for layer in self.model.decoder]
        indices = []
        for logit in logits:
            probability = torch.softmax(logit / max(temperature, 0.05), -1).numpy()
            indices.append(int(rng.choice(len(probability), p=probability)) if stochastic else int(probability.argmax()))
        return self.space.from_indices(indices)

    def checkpoint(self):
        return {"state_dict": self.model.state_dict(), "version": self.version, "y_mean": self.y_mean,
                "y_std": self.y_std, "performance_fitted": self.performance_fitted,
                "generator_state": self.generator.get_state()}
