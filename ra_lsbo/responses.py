"""Independent objective/constraint GPs using the established LSBO fit routine."""

import torch

from .features import history_identity, validate_history
from .training import training_rows


class ResponseHeads:
    def __init__(self, rows, features, seed=0, steps=45, use_prior=True):
        validate_history(rows)
        if type(steps) is not int or steps < 1:
            raise ValueError("Positive integer GP steps required")
        if (features.training_indices != [r["evaluation"] for r in rows]
                or features.history_id != history_identity(rows)):
            raise ValueError("Representation and response heads must use the same prefix")
        self.features, self.snapshot_id = features, features.snapshot_id
        self.models, self.diagnostics = {}, {}
        rep = features.representation
        n = len(features.space.continuous)
        with torch.no_grad():
            z, _ = rep.model.encode(rep.grammar)
        self.offset = torch.cat([torch.zeros(2*n), z.min(0).values - .5]).double()
        upper = torch.cat([torch.ones(2*n), z.max(0).values + .5]).double()
        self.scale = (upper - self.offset).clamp_min(1e-8)
        tables = training_rows(rows)
        datasets = {"objective": tables["objective"],
                    **{"constraint:" + name: table for name, table in tables["constraints"].items()}}
        for name, table in datasets.items():
            diagnostics = {"samples": len(table), "training_indices": [r["evaluation"] for r in table],
                           "representation_snapshot": self.snapshot_id, "calibrated": False}
            self.diagnostics[name] = diagnostics
            if len(table) < 3:
                diagnostics["status"] = "unavailable_insufficient_observations"
                continue
            x = (features.transform([r["candidate"] for r in table]) - self.offset) / self.scale
            sign = -1. if name == "objective" else 1.
            values = sign * torch.tensor([r["target"] for r in table], dtype=torch.double)
            center = float(values.mean())
            scale = float(values.std().clamp_min(1e-6))
            target = ((values - center) / scale).unsqueeze(-1)
            prior = None
            base = features.modules["surrogate"]
            if name == "objective" and use_prior and rep.performance_fitted:
                prior = base.FrozenPerformanceMean(rep, center, scale, self.offset, self.scale)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                gp = base.fit_gp(x, target, steps, mean_module=prior, name=name)
            self.models[name] = (gp, center, scale, sign)
            diagnostics.update(status="fitted_raw", prior_used=prior is not None,
                               fit=gp.fit_diagnostics)

    def predict(self, candidates):
        if self.features.snapshot_id != self.snapshot_id:
            raise RuntimeError("Representation changed; refit all response heads")
        x = (self.features.transform(candidates) - self.offset) / self.scale
        result = {}
        with torch.no_grad():
            for name in self.diagnostics:
                if name not in self.models:
                    result[name] = None
                    continue
                gp, center, scale, sign = self.models[name]
                post = gp.posterior(x)
                mean = sign * (post.mean.squeeze(-1) * scale + center)
                sd = post.variance.squeeze(-1).clamp_min(0).sqrt() * scale
                if not torch.isfinite(mean).all() or not torch.isfinite(sd).all():
                    raise FloatingPointError("Invalid GP prediction")
                output = {"mean": mean.tolist(), "std": sd.tolist()}
                if name != "objective":
                    z = -mean / sd.clamp_min(1e-12)
                    output["p_satisfied_raw"] = torch.distributions.Normal(0., 1.).cdf(z).tolist()
                result[name] = output
        return result
