"""Masked ICM constraint GP using BoTorch's established MultiTaskGP."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path

from botorch.models.multitask import MultiTaskGP
import gpytorch
import torch

from .budget import digest
from .features import history_identity, validate_history
from .training import training_rows


@dataclass(frozen=True)
class JointConstraintSettings:
    steps: int = 60
    rank: int = 2
    learning_rate: float = .05
    minimum_observations: int = 3

    def __post_init__(self):
        if (type(self.steps) is not int or self.steps < 1 or type(self.rank) is not int
                or self.rank < 0 or not 0 < self.learning_rate <= 1
                or type(self.minimum_observations) is not int or self.minimum_observations < 3):
            raise ValueError("Invalid multitask GP settings")


class JointConstraintPosterior:
    def __init__(self, rows, features, analytical_constraints, seed=0, settings=None):
        validate_history(rows)
        features.assert_frozen()
        if (features.history_id != history_identity(rows)
                or features.training_indices != [r["evaluation"] for r in rows]):
            raise ValueError("Constraint GP and representation histories differ")
        self.features, self.settings = features, settings or JointConstraintSettings()
        self.seed, self.snapshot_id = seed, features.snapshot_id
        self.history_id, self.analytical_constraints = history_identity(rows), list(analytical_constraints)
        self.model, self.diagnostics = None, {}
        tables = training_rows(rows)["constraints"]
        tables = {name: table for name, table in tables.items() if name not in analytical_constraints}
        for name, table in sorted(tables.items()):
            self.diagnostics[name] = {"samples": len(table),
                                      "training_indices": [r["evaluation"] for r in table],
                                      "supported": len(table) >= self.settings.minimum_observations}
        self.names = [name for name in sorted(tables) if self.diagnostics[name]["supported"]]
        self.unsupported = [name for name in sorted(tables) if name not in self.names]
        all_x = features.transform([r["candidate"] for r in rows]).double()
        self.offset = all_x.min(0).values
        self.scale = (all_x.max(0).values-self.offset).clamp_min(1.)
        self.centers, self.scales = [], []
        x_parts, y_parts = [], []
        for task, name in enumerate(self.names):
            table = tables[name]
            values = torch.tensor([r["target"] for r in table], dtype=torch.double)
            center, scale = float(values.mean()), float(values.std().clamp_min(1e-6))
            self.centers.append(center)
            self.scales.append(scale)
            x = self.inputs([r["candidate"] for r in table])
            x_parts.append(torch.cat([x, torch.full((len(table), 1), task, dtype=torch.double)], dim=1))
            y_parts.append(((values-center)/scale).unsqueeze(-1))
        self.losses = []
        if self.names:
            train_x, train_y = torch.cat(x_parts), torch.cat(y_parts)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.model = MultiTaskGP(
                    train_x, train_y, task_feature=-1, rank=min(self.settings.rank, len(self.names)),
                    outcome_transform=None).double()
                with torch.no_grad():
                    self.model.likelihood.noise = .01
                    self.model.task_covar_module.raw_var.zero_()
                    self.model.task_covar_module.covar_factor.mul_(.05)
                self.model.train()
                self.model.likelihood.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                for _ in range(self.settings.steps):
                    optimizer.zero_grad()
                    loss = -mll(self.model(train_x), train_y.squeeze(-1))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite joint constraint training loss")
                    loss.backward()
                    optimizer.step()
                    self.losses.append(float(loss.detach()))
                self.model.eval()
                self.model.likelihood.eval()
        self.model_id = self.identity()

    def inputs(self, candidates):
        if self.features.snapshot_id != self.snapshot_id:
            raise RuntimeError("Representation changed; refit the joint constraint GP")
        return (self.features.transform(candidates).double()-self.offset)/self.scale

    def identity(self):
        state = {} if self.model is None else {
            name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, value in self.model.state_dict().items()}
        return digest({
            "schema": "masked_icm_constraints_v1", "history": self.history_id,
            "representation": self.snapshot_id, "settings": asdict(self.settings),
            "seed": self.seed, "names": self.names, "diagnostics": self.diagnostics,
            "offset": self.offset.tolist(), "input_scale": self.scale.tolist(),
            "centers": self.centers, "scales": self.scales, "state": state,
            "botorch": importlib.metadata.version("botorch"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })

    def require_supported(self):
        if self.model is None or self.unsupported:
            raise ValueError("Joint engineering feasibility unsupported: "+str(self.unsupported))

    def pointwise_moments(self, candidates):
        """Latent-function moments; excludes observation noise and physical uncertainty."""
        self.require_supported()
        with torch.no_grad():
            post = self.model.posterior(self.inputs(candidates).unsqueeze(-2))
            scales = torch.tensor(self.scales, dtype=torch.double)
            centers = torch.tensor(self.centers, dtype=torch.double)
            mean = post.mean.squeeze(-2)*scales+centers
            covariance = post.mvn.covariance_matrix*scales.unsqueeze(-1)*scales.unsqueeze(-2)
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise FloatingPointError("Nonfinite joint constraint prediction")
        return mean, covariance

    def sample(self, candidates, seed, samples=1):
        """Joint across both candidates and constraint heads, in physical units."""
        self.require_supported()
        if type(samples) is not int or samples < 1:
            raise ValueError("Positive integer posterior sample count required")
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(seed)
            draws = self.model.posterior(self.inputs(candidates)).rsample(torch.Size([samples]))
            return (draws*torch.tensor(self.scales, dtype=torch.double)
                    +torch.tensor(self.centers, dtype=torch.double))

    def feasibility(self, candidates, seed, samples=4096):
        draws = self.sample(candidates, seed, samples)
        probability = (draws <= 0).all(-1).double().mean(0)
        return {"constraint_names": self.names, "conditional_joint_raw": probability.tolist(),
                "monte_carlo_samples": samples, "sample_seed": seed,
                "calibration_status": "not_applied", "probability_guarantee": False,
                "interpretation": "model-conditional latent constraint probability; not physical reliability"}

    def save_checkpoint(self, path):
        with Path(path).open("xb") as stream:
            torch.save({
                "schema": "masked_icm_constraints_v1", "model_id": self.model_id,
                "history_id": self.history_id, "snapshot_id": self.snapshot_id,
                "settings": asdict(self.settings), "seed": self.seed,
                "analytical_constraints": self.analytical_constraints, "losses": self.losses,
                "state": None if self.model is None else self.model.state_dict(),
            }, stream)

    @classmethod
    def load_checkpoint(cls, path, features, rows):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["schema"] != "masked_icm_constraints_v1"
                or saved["history_id"] != history_identity(rows)
                or saved["snapshot_id"] != features.snapshot_id):
            raise ValueError("Joint GP checkpoint history or representation differs")
        model = cls(rows, features, saved["analytical_constraints"], saved["seed"],
                    JointConstraintSettings(**dict(saved["settings"], steps=1)))
        model.settings = JointConstraintSettings(**saved["settings"])
        if (model.model is None) != (saved["state"] is None):
            raise ValueError("Joint GP checkpoint support differs")
        if model.model is not None:
            model.model.load_state_dict(saved["state"], strict=True)
            model.model.eval()
        model.losses = saved["losses"]
        model.model_id = model.identity()
        if model.model_id != saved["model_id"]:
            raise ValueError("Joint GP checkpoint identity differs")
        return model
