"""Censored completion learning with variational probit Gaussian processes."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import gpytorch
import numpy as np
import torch

from .adaptive_risk import weights_digest
from .budget import digest
from .features import history_identity, validate_history
from .records import HIDDEN_MODES


@dataclass(frozen=True)
class PosteriorSettings:
    steps: int = 100
    inducing: int = 64
    learning_rate: float = .03
    view: str = "paper_gp"
    mechanisms: bool = True

    def __post_init__(self):
        if (type(self.steps) is not int or self.steps < 1
                or type(self.inducing) is not int or self.inducing < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or self.view not in ("paper_gp", "hierarchical_onehot")
                or type(self.mechanisms) is not bool):
            raise ValueError("Invalid posterior classifier settings")


class ProbitGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing):
        variational = gpytorch.variational.CholeskyVariationalDistribution(len(inducing))
        strategy = gpytorch.variational.VariationalStrategy(
            self, inducing, variational, learn_inducing_locations=False)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=inducing.shape[1],
            lengthscale_prior=gpytorch.priors.GammaPrior(3., 3.))
        # Fixed unit prior variance avoids training an all-success region's
        # output variance down to zero everywhere.
        self.covar_module.lengthscale = 1.

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def spread_indices(x, count):
    """Deterministic farthest-point inducing subset; labels do not select it."""
    chosen = [0]
    nearest = torch.full((len(x),), torch.inf, dtype=x.dtype)
    for _ in range(min(count, len(x))-1):
        nearest = torch.minimum(nearest, ((x-x[chosen[-1]])**2).sum(1))
        nearest[chosen] = -1.
        chosen.append(int(nearest.argmax()))
    return chosen


class CompletionPosterior:
    def __init__(self, rows, features, seed=0, settings=None):
        validate_history(rows)
        features.assert_frozen()
        if features.history_id != history_identity(rows):
            raise ValueError("Completion model and representation history differ")
        self.features, self.snapshot_id = features, features.snapshot_id
        self.settings, self.seed = settings or PosteriorSettings(), seed
        self.training_indices = [r["evaluation"] for r in rows]
        self.history_id = history_identity(rows)
        x = features.transform_view([r["candidate"] for r in rows], self.settings.view).double()
        # The continuous inputs already use engineering bounds. Estimate bounds
        # only for learned latent coordinates; never use test/pool extrema.
        self.offset = torch.zeros(x.shape[1], dtype=torch.double)
        self.scale = torch.ones(x.shape[1], dtype=torch.double)
        if self.settings.view == "paper_gp":
            n = features.settings.latent_dim
            self.offset[-n:] = x[:, -n:].min(0).values-.5
            self.scale[-n:] = (x[:, -n:].max(0).values+.5)-self.offset[-n:]
        x = (x-self.offset)/self.scale
        labels = {"completion": [r["risk"]["hidden_success"] for r in rows]}
        if self.settings.mechanisms:
            labels.update({name: [r["risk"]["hidden_labels"][name] for r in rows]
                           for name in HIDDEN_MODES})
        self.models, self.diagnostics = {}, {}
        for number, (name, values) in enumerate(labels.items()):
            observed = [i for i, value in enumerate(values) if value is not None]
            y = torch.tensor([values[i] for i in observed], dtype=torch.double)
            diagnostic = {
                "training_indices": [rows[i]["evaluation"] for i in observed],
                "positive": int(y.sum()), "negative": len(y)-int(y.sum()),
                "unknown": len(rows)-len(y), "probability_guarantee": False}
            self.diagnostics[name] = diagnostic
            if not observed or (name != "completion" and len(set(y.tolist())) < 2):
                diagnostic["status"] = "unsupported_no_observations_or_no_failure_contrast"
                continue
            train = x[observed]
            indices = spread_indices(train, self.settings.inducing)
            with torch.random.fork_rng():
                torch.manual_seed(seed+7919*number)
                model = ProbitGP(train[indices].clone()).double()
                likelihood = gpytorch.likelihoods.BernoulliLikelihood().double()
                model.train()
                likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.settings.learning_rate)
                objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y))
                for _ in range(self.settings.steps):
                    optimizer.zero_grad()
                    loss = -objective(model(train), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite completion GP objective")
                    loss.backward()
                    optimizer.step()
                model.eval()
                likelihood.eval()
            self.models[name] = model
            diagnostic.update(status="variational_probit_gp", last_loss=float(loss.detach()),
                              inducing_training_indices=[rows[observed[i]]["evaluation"]
                                                         for i in indices])
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.weight_id = weights_digest([self.models[n] for n in sorted(self.models)])
        self.model_id = digest(self._identity())

    def _identity(self):
        return {"history": self.history_id, "snapshot": self.snapshot_id,
                "settings": asdict(self.settings), "seed": self.seed,
                "weights": self.weight_id, "offset": self.offset.tolist(),
                "scale": self.scale.tolist(), "source": self.source_sha256,
                "diagnostics": self.diagnostics}

    def assert_frozen(self):
        self.features.assert_frozen()
        if (self.features.snapshot_id != self.snapshot_id
                or self.weight_id != weights_digest([self.models[n] for n in sorted(self.models)])
                or self.model_id != digest(self._identity())):
            raise RuntimeError("Completion posterior changed after fitting")

    def inputs(self, candidates):
        self.assert_frozen()
        return (self.features.transform_view(candidates, self.settings.view).double()
                - self.offset)/self.scale

    def predict(self, candidates):
        x = self.inputs(candidates)
        outputs = {}
        with torch.no_grad():
            for name, diagnostic in self.diagnostics.items():
                if name not in self.models:
                    outputs[name] = None
                    continue
                latent = self.models[name](x)
                probability = torch.distributions.Normal(0., 1.).cdf(
                    latent.mean/(1.+latent.variance).sqrt())
                outputs[name] = {"probability": probability.tolist(),
                                 "latent_mean": latent.mean.tolist(),
                                 "latent_variance": latent.variance.tolist()}
        return {"heads": outputs, "model_id": self.model_id,
                "unsupported": [n for n, d in self.diagnostics.items()
                                if not d["status"].startswith("variational")],
                "probability_guarantee": False}

    def sample_completion(self, candidates, seed):
        x = self.inputs(candidates)
        if "completion" not in self.models:
            raise ValueError("No observed completion model; use explicit bootstrap")
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(seed)
            latent = self.models["completion"](x).rsample()
            probability = torch.distributions.Normal(0., 1.).cdf(latent)
            # A shared latent draw preserves dependence across candidate inputs.
            success = torch.rand(len(x), dtype=torch.double) < probability
        return success.numpy(), probability.numpy()

    def save_checkpoint(self, path):
        self.assert_frozen()
        with Path(path).open("xb") as stream:
            torch.save({"identity": self._identity(),
                        "states": {n: m.state_dict() for n, m in self.models.items()}}, stream)

    @classmethod
    def load_checkpoint(cls, path, features, rows):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        identity = payload["identity"]
        result = cls(rows, features, identity["seed"],
                     PosteriorSettings(**dict(identity["settings"], steps=1)))
        # Rebuilding shapes is deterministic; do not refit for recovery.
        result.settings = PosteriorSettings(**identity["settings"])
        for name, state in payload["states"].items():
            result.models[name].load_state_dict(state, strict=True)
            result.models[name].eval()
        result.diagnostics = identity["diagnostics"]
        result.weight_id = weights_digest([result.models[n] for n in sorted(result.models)])
        result.model_id = digest(result._identity())
        if result._identity() != identity:
            raise ValueError("Completion checkpoint identity differs")
        return result
