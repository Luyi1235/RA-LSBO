"""Paper-aligned structured encoding, isolated from the original LSBO baseline."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .budget import digest
from .features import LatentFeatures, history_identity, load_base, validate_history


VIEWS = ("latent", "paper_gp", "hierarchical_onehot", "plain_onehot", "continuous_mask")


@dataclass(frozen=True)
class EncoderSettings:
    latent_dim: int = 4
    width: int = 48
    embedding_dim: int = 4
    batch_size: int = 128
    initial_epochs: int = 60
    update_epochs: int = 20
    learning_rate: float = .001
    update_learning_rate: float = .0004
    mmd_weight: float = .05
    performance_weight: float = 1.
    entropy_weight: float = .01
    warmup_fraction: float = .25
    ramp_fraction: float = .5
    performance_schedule: str = "staged"
    joint_view: str = "paper_gp"

    def __post_init__(self):
        for name in ("latent_dim", "width", "embedding_dim", "batch_size",
                     "initial_epochs", "update_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer encoder settings required")
        for name in ("learning_rate", "update_learning_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Positive finite learning rates required")
        for name in ("mmd_weight", "performance_weight", "entropy_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("Finite nonnegative loss weights required")
        if (not 0 <= self.warmup_fraction < 1 or not 0 < self.ramp_fraction <= 1
                or self.warmup_fraction+self.ramp_fraction > 1
                or self.performance_schedule not in ("staged", "constant")
                or self.joint_view not in ("latent", "paper_gp")):
            raise ValueError("Invalid encoder schedule or joint feature layout")

    def weights(self, epoch, epochs, incremental=False):
        if type(epoch) is not int or not 0 <= epoch < epochs:
            raise ValueError("Epoch outside training schedule")
        fraction = epoch/max(1, epochs-1)
        ramp = np.clip((fraction-self.warmup_fraction)/self.ramp_fraction, 0., 1.)
        gamma = self.performance_weight * (
            1. if incremental or self.performance_schedule == "constant" else float(ramp))
        entropy = 0. if incremental else self.entropy_weight*(1.-fraction)
        return {"mmd": self.mmd_weight, "performance": gamma, "entropy": entropy}


def hierarchy_active(space, variable, discrete):
    by_name = {v.name: v for v in space.discrete}
    for parent, allowed in variable.active_when.items():
        if (not hierarchy_active(space, by_name[parent], discrete)
                or discrete[parent] not in allowed):
            return False
    return True


def hierarchy_space(modules, variables):
    # The inherited schema validator already requires parents before children.
    cls = type("HierarchicalDesignSpace", (modules["space"].DesignSpace,),
               {"active": hierarchy_active})
    space = cls.from_dict({"variables": variables})
    if not space.discrete:
        raise ValueError("Structured encoder requires at least one discrete variable")
    return space


class LatentSlice(nn.Module):
    """Expose a full joint-input prior whose learned head sees only z."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.in_features = input_dim
        self.latent_dim = latent_dim

    def forward(self, value):
        if value.shape[-1] != self.in_features:
            raise ValueError("Performance prior received the wrong feature layout")
        return value[..., -self.latent_dim:]


class PaperStructuredVAE(nn.Module):
    def __init__(self, space, settings):
        super().__init__()
        self.space, self.settings, self.latent_dim = space, settings, settings.latent_dim
        self.embeddings = nn.ModuleDict({
            v.name: nn.Embedding(len(v.values), settings.embedding_dim)
            for v in space.discrete if v.kind == "nominal"})
        dimensions = sum(3 if v.kind == "ordinal" else settings.embedding_dim+1
                         for v in space.discrete)
        self.encoder = nn.Sequential(nn.Linear(dimensions, settings.width), nn.SiLU(),
                                     nn.Linear(settings.width, settings.width), nn.SiLU())
        self.mu, self.logvar = (nn.Linear(settings.width, settings.latent_dim) for _ in range(2))
        self.decoder = nn.ModuleList([nn.Linear(settings.latent_dim, len(v.values))
                                      for v in space.discrete])
        n = len(space.continuous)*(2 if settings.joint_view == "latent" else 1)
        self.performance = nn.Sequential(
            LatentSlice(n+settings.latent_dim, settings.latent_dim),
            nn.Linear(settings.latent_dim, settings.width), nn.SiLU(),
            nn.Linear(settings.width, 1))

    def discrete_mask(self, indices):
        if indices.ndim != 2 or indices.shape[1] != len(self.space.discrete):
            raise ValueError("Discrete indices must match the schema")
        masks, positions = [], {v.name: i for i, v in enumerate(self.space.discrete)}
        for i, variable in enumerate(self.space.discrete):
            if ((indices[:, i] < 0) | (indices[:, i] >= len(variable.values))).any():
                raise ValueError("Discrete index outside its declared categories")
            mask = torch.ones(indices.shape[0], dtype=torch.double, device=indices.device)
            for parent, allowed in variable.active_when.items():
                j = positions[parent]
                valid = torch.zeros_like(mask)
                for value in allowed:
                    valid += (indices[:, j] == self.space.discrete[j].values.index(value)).double()
                mask *= masks[j]*valid
            masks.append(mask)
        return torch.stack(masks, -1)

    def encode(self, indices):
        masks, parts = self.discrete_mask(indices), []
        for i, variable in enumerate(self.space.discrete):
            mask = masks[:, i:i+1]
            if variable.kind == "ordinal":
                unit = indices[:, i:i+1].double()/(len(variable.values)-1)
                embedding = torch.cat([unit, torch.sin(torch.pi*unit)], -1)
            else:
                embedding = self.embeddings[variable.name](indices[:, i])
            parts.extend([embedding*mask, mask])
        hidden = self.encoder(torch.cat(parts, -1))
        return self.mu(hidden), self.logvar(hidden).clamp(-8., 4.)

    def performance_from_latent(self, latent):
        return self.performance[1:](latent).squeeze(-1)

    def loss_terms(self, indices, generator):
        mu, logvar = self.encode(indices)
        z = mu+(logvar*.5).exp()*torch.randn(mu.shape, generator=generator, dtype=torch.double)
        masks = self.discrete_mask(indices)
        reconstruction, entropy = [], []
        for i, layer in enumerate(self.decoder):
            logits = layer(z)
            log_probability = F.log_softmax(logits, -1)
            reconstruction.append(F.cross_entropy(logits, indices[:, i], reduction="none")*masks[:, i])
            entropy.append(-(log_probability.exp()*log_probability).sum(-1)*masks[:, i])
        reference = torch.randn(z.shape, generator=generator, dtype=torch.double)

        def kernel(a, b):
            distance = torch.cdist(a, b).square()/self.latent_dim
            return sum(torch.exp(-distance/scale) for scale in (.5, 1., 2.))/3

        return {
            "reconstruction": torch.stack(reconstruction, -1).sum(-1).mean(),
            "mmd": kernel(z, z).mean()+kernel(reference, reference).mean()-2*kernel(z, reference).mean(),
            "entropy": torch.stack(entropy, -1).sum(-1).mean()}


class StructuredRepresentation:
    def __init__(self, space, settings, seed=0):
        self.space, self.settings = space, settings
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = PaperStructuredVAE(space, settings).double()
        self.generator = torch.Generator().manual_seed(seed+1)
        self.grammar = torch.tensor([space.indices(d) for d in space.configurations()], dtype=torch.long)
        self.version, self.y_mean, self.y_std = 0, 0., 1.
        self.performance_fitted = False
        self.training_trace = []

    def _indices(self, candidates):
        return torch.tensor([self.space.indices(c.discrete) for c in candidates], dtype=torch.long)

    def base_features(self, candidates):
        masks = np.asarray([self.space.mask(c.discrete) for c in candidates])
        units = np.asarray([self.space.unit(c) for c in candidates])
        columns = [units*masks]
        if self.settings.joint_view == "latent":
            columns.append(masks)
        return torch.tensor(np.hstack(columns), dtype=torch.double)

    def features(self, candidates):
        with torch.no_grad():
            mu, _ = self.model.encode(self._indices(candidates))
        return torch.cat([self.base_features(candidates), mu], -1)

    def latent(self, discrete):
        with torch.no_grad():
            mu, _ = self.model.encode(torch.tensor([self.space.indices(discrete)], dtype=torch.long))
        return mu[0]

    def fit(self, observations, epochs=None, incremental=False):
        settings = self.settings
        epochs = epochs if epochs is not None else (
            settings.update_epochs if incremental else settings.initial_epochs)
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Positive training epochs required")
        usable = [o for o in observations if o.result.has_objective()]
        target, perf_indices = None, None
        if len(usable) >= 3:
            energy = np.asarray([o.result.objective for o in usable])
            self.y_mean, self.y_std = float(energy.mean()), float(max(energy.std(), 1e-8))
            target = torch.tensor((energy-self.y_mean)/self.y_std, dtype=torch.double)
            perf_indices = self._indices([o.candidate for o in usable])
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=settings.update_learning_rate if incremental else settings.learning_rate)
        self.training_trace = []
        for epoch in range(epochs):
            weights, totals = settings.weights(epoch, epochs, incremental), Counter()
            batches = 0
            for batch in torch.randperm(len(self.grammar), generator=self.generator).split(settings.batch_size):
                optimizer.zero_grad()
                terms = self.model.loss_terms(self.grammar[batch], self.generator)
                performance = torch.zeros((), dtype=torch.double)
                if target is not None and weights["performance"] > 0:
                    chosen = torch.randperm(len(usable), generator=self.generator)[:settings.batch_size]
                    latent, _ = self.model.encode(perf_indices[chosen])
                    performance = F.mse_loss(self.model.performance_from_latent(latent), target[chosen])
                loss = (terms["reconstruction"]+weights["mmd"]*terms["mmd"]
                        +weights["performance"]*performance-weights["entropy"]*terms["entropy"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite structured encoder loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in self.model.parameters()):
                    raise FloatingPointError("Nonfinite encoder gradient")
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()
                for name, value in dict(terms, performance=performance, total=loss).items():
                    totals[name] += float(value.detach())
                batches += 1
            self.training_trace.append({
                "epoch": epoch+1, "weights": weights,
                "losses": {name: value/batches for name, value in totals.items()},
                "incremental": incremental})
        self.model.eval()
        self.performance_fitted = (target is not None and
                                   any(r["weights"]["performance"] > 0 for r in self.training_trace))
        self.version += 1

    def decode(self, z, rng, temperature=1., stochastic=False):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive finite decoding temperature required")
        z = torch.as_tensor(z, dtype=torch.double)
        if z.shape != (self.settings.latent_dim,) or not torch.isfinite(z).all():
            raise ValueError("Invalid latent vector")
        result = {}
        with torch.no_grad():
            for variable, layer in zip(self.space.discrete, self.model.decoder):
                if not self.space.active(variable, result):
                    result[variable.name] = variable.default
                    continue
                probability = torch.softmax(layer(z)/temperature, -1).numpy()
                index = (int(rng.choice(len(probability), p=probability)) if stochastic
                         else int(probability.argmax()))
                result[variable.name] = variable.values[index]
        return self.space.canonical(type(self.space.default())({}, result)).discrete


def state_digest(representation):
    state = hashlib.sha256()
    for name, tensor in sorted(representation.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        state.update(digest({"name": name, "shape": value.shape, "dtype": str(value.dtype)}).encode())
        state.update(value.tobytes())
    state.update(representation.generator.get_state().numpy().tobytes())
    state.update(representation.grammar.numpy().tobytes())
    state.update(digest({"version": representation.version, "mean": representation.y_mean,
                         "std": representation.y_std,
                         "performance_fitted": representation.performance_fitted}).encode())
    return state.hexdigest()


class StructuredFeatures(LatentFeatures):
    """Fit/reset or append-only incremental update; snapshots bind every feature view."""

    def __init__(self, base_project, variables, seed=0, settings=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("Nonnegative integer encoder seed required")
        self.settings, self.seed = settings or EncoderSettings(), seed
        self.modules, self.base_hashes = load_base(base_project)
        self.space = hierarchy_space(self.modules, variables)
        self.variables = [asdict(variable) for variable in self.space.variables]
        self.representation = StructuredRepresentation(self.space, self.settings, seed)
        self.training_indices = self.history_id = self.snapshot_id = None
        self._row_ids = []
        self._sealed_state = None
        self.source_sha256 = {
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str(Path(__file__).with_name("features.py").resolve()):
                hashlib.sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest()}

    def layout(self, view=None):
        view = view or self.settings.joint_view
        if view not in VIEWS:
            raise ValueError("Unknown structured feature view")
        columns = []
        for variable in self.space.continuous:
            columns.append(("unit:" if view == "plain_onehot" else "masked_unit:")+variable.name)
        if view not in ("plain_onehot", "paper_gp"):
            columns.extend("active:"+v.name for v in self.space.continuous)
        if view in ("latent", "paper_gp"):
            columns.extend("z:"+str(i) for i in range(self.settings.latent_dim))
        elif view in ("plain_onehot", "hierarchical_onehot"):
            for variable in self.space.discrete:
                columns.extend("category:"+variable.name+":"+str(i) for i in range(len(variable.values)))
                if view == "hierarchical_onehot":
                    columns.append("active:"+variable.name)
        return {"view": view, "dimensions": len(columns), "columns": columns,
                "latent_dimensions": self.settings.latent_dim if view in ("latent", "paper_gp") else 0}

    def _observations(self, rows):
        return [self.modules["records"].Observation(
            row["evaluation"], self.candidate(row["candidate"]),
            self.modules["records"].Evaluation(
                objective=row["risk"]["objective"], solver_success=row["risk"]["objective_observed"],
                constraints=row["risk"]["constraints"], status=row["risk"]["raw_status"]),
            row.get("phase", "offline")) for row in rows]

    def fit(self, rows, epochs=None):
        validate_history(rows)
        if not rows:
            raise ValueError("Nonempty encoder training prefix required")
        version = self.representation.version
        replacement = StructuredRepresentation(self.space, self.settings, self.seed)
        replacement.version = version
        replacement.fit(self._observations(rows), epochs=epochs)
        self.representation = replacement
        self._seal(rows)
        return self

    def update(self, rows, epochs=None):
        self.assert_frozen()
        validate_history(rows)
        row_ids = [history_identity([row]) for row in rows]
        if len(rows) <= len(self._row_ids) or row_ids[:len(self._row_ids)] != self._row_ids:
            raise ValueError("Incremental update requires an unchanged, strictly extended prefix")
        replacement = deepcopy(self.representation)
        replacement.fit(self._observations(rows), epochs=epochs, incremental=True)
        self.representation = replacement
        self._seal(rows)
        return self

    def _seal(self, rows):
        self.training_indices = [row["evaluation"] for row in rows]
        self.history_id, self._row_ids = history_identity(rows), [history_identity([row]) for row in rows]
        self.representation_version = self.representation.version
        self.objective_indices = [r["evaluation"] for r in rows if r["risk"]["objective_observed"]]
        self._sealed_state = state_digest(self.representation)
        self.snapshot_id = digest(self._identity())

    def _identity(self):
        return {
            "schema": "ra_structured_features_v1", "variables": self.variables,
            "settings": asdict(self.settings), "seed": self.seed,
            "training_indices": self.training_indices, "history_id": self.history_id,
            "row_identities": self._row_ids, "objective_indices": self.objective_indices,
            "version": self.representation.version, "model_state_sha256": self._sealed_state,
            "base_source_sha256": self.base_hashes, "encoder_source_sha256": self.source_sha256,
            "training_trace": self.representation.training_trace,
            "layouts": {view: self.layout(view) for view in VIEWS}}

    def assert_frozen(self):
        if self.snapshot_id is None:
            raise RuntimeError("Structured encoder is not fitted")
        if (digest([asdict(v) for v in self.space.variables]) != digest(self.variables)
                or self.representation.version != self.representation_version
                or state_digest(self.representation) != self._sealed_state
                or digest(self._identity()) != self.snapshot_id):
            raise RuntimeError("Encoder changed outside its snapshot; refit downstream models")

    def transform(self, candidates):
        return self.transform_view(candidates, self.settings.joint_view)

    def transform_view(self, candidates, view="latent"):
        self.assert_frozen()
        dimensions = self.layout(view)["dimensions"]
        canonical = [self.candidate(raw) for raw in candidates]
        if not canonical:
            return torch.empty((0, dimensions), dtype=torch.double)
        units = np.asarray([self.space.unit(c) for c in canonical])
        masks = np.asarray([self.space.mask(c.discrete) for c in canonical])
        parts = [units if view == "plain_onehot" else units*masks]
        if view not in ("plain_onehot", "paper_gp"):
            parts.append(masks)
        if view in ("latent", "paper_gp"):
            with torch.no_grad():
                latent, _ = self.representation.model.encode(self.representation._indices(canonical))
            parts.append(latent.numpy())
        elif view in ("plain_onehot", "hierarchical_onehot"):
            for variable in self.space.discrete:
                active = np.asarray([self.space.active(variable, c.discrete) for c in canonical], dtype=float)
                indicator = np.zeros((len(canonical), len(variable.values)))
                indicator[np.arange(len(canonical)),
                          [variable.values.index(c.discrete[variable.name]) for c in canonical]] = 1.
                parts.append(indicator*active[:, None] if view == "hierarchical_onehot" else indicator)
                if view == "hierarchical_onehot":
                    parts.append(active[:, None])
        result = torch.tensor(np.hstack(parts), dtype=torch.double)
        if result.shape != (len(candidates), dimensions) or not torch.isfinite(result).all():
            raise FloatingPointError("Invalid structured features")
        return result

    def save_checkpoint(self, path):
        self.assert_frozen()
        metadata = dict(self._identity(), snapshot_id=self.snapshot_id,
                        training_trace=self.representation.training_trace,
                        y_mean=self.representation.y_mean, y_std=self.representation.y_std,
                        performance_fitted=self.representation.performance_fitted)
        with Path(path).open("xb") as stream:
            torch.save({"metadata_json": json.dumps(metadata, sort_keys=True, allow_nan=False),
                        "model": self.representation.model.state_dict(),
                        "generator": self.representation.generator.get_state()}, stream)
        return self.snapshot_id

    @classmethod
    def load_checkpoint(cls, path, base_project, expected_history=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = json.loads(payload["metadata_json"])
        if metadata.get("schema") != "ra_structured_features_v1":
            raise ValueError("Unsupported structured encoder checkpoint")
        features = cls(base_project, metadata["variables"], metadata["seed"],
                       EncoderSettings(**metadata["settings"]))
        if (metadata["base_source_sha256"] != features.base_hashes
                or metadata["encoder_source_sha256"] != features.source_sha256):
            raise ValueError("Checkpoint source identity changed")
        if expected_history is not None:
            validate_history(expected_history)
            if history_identity(expected_history) != metadata["history_id"]:
                raise ValueError("Checkpoint training history differs")
        rep = features.representation
        rep.model.load_state_dict(payload["model"], strict=True)
        rep.generator.set_state(payload["generator"])
        rep.version, rep.y_mean, rep.y_std = metadata["version"], metadata["y_mean"], metadata["y_std"]
        rep.performance_fitted = metadata["performance_fitted"]
        rep.training_trace = metadata["training_trace"]
        rep.model.eval()
        features.training_indices, features.history_id = metadata["training_indices"], metadata["history_id"]
        features._row_ids, features.objective_indices = metadata["row_identities"], metadata["objective_indices"]
        features.representation_version = rep.version
        features._sealed_state, features.snapshot_id = metadata["model_state_sha256"], metadata["snapshot_id"]
        features.assert_frozen()
        return features


