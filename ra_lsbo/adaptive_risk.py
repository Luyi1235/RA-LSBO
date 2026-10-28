"""Experimental joint terminal-event ensembles with partially observed labels."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F

from .budget import digest
from .features import history_identity, validate_history
from .records import HIDDEN_MODES


EVENTS = ("feasible", "explicit_infeasible")+HIDDEN_MODES


def terminal_sets(rows, aggregate=False):
    """Unknown downstream stages never become negative terminal labels."""
    allowed = np.zeros((len(rows), 3 if aggregate else 7), dtype=bool)
    for i, row in enumerate(rows):
        risk = row["risk"]
        failures = [j for j, name in enumerate(HIDDEN_MODES) if risk["hidden_labels"][name] == 1]
        if risk["state"] == "hidden_failure" and len(failures) == 1:
            allowed[i, 2 if aggregate else failures[0]+2] = True
        elif risk["objective_observed"]:
            if risk["feasible_label"] is True:
                allowed[i, 0] = True
            elif risk["feasible_label"] is False:
                allowed[i, 1] = True
            else:
                allowed[i, :2] = True
    return allowed


def partial_event_loss(logits, allowed):
    known = allowed.any(1)
    if not known.any():
        return logits.sum()*0.
    log_probability = F.log_softmax(logits[known], dim=1)
    return -torch.logsumexp(log_probability.masked_fill(~allowed[known], -torch.inf), dim=1).mean()


def supervised_metric_loss(embedding, allowed):
    known = allowed.sum(1) == 1
    if int(known.sum()) < 3:
        return embedding.sum()*0.
    z, labels = F.normalize(embedding[known], dim=1), allowed[known].long().argmax(1)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    usable = positive.any(1)
    if not usable.any():
        return embedding.sum()*0.
    similarity = (z @ z.T/.2).masked_fill(diagonal, -1e9)
    log_probability = similarity-torch.logsumexp(similarity, dim=1, keepdim=True)
    return -(log_probability*positive).sum(1)[usable].div(positive.sum(1)[usable]).mean()


def entropy_decomposition(member_probabilities):
    p = np.asarray(member_probabilities, dtype=float)
    if (p.ndim != 3 or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(-1), 1., atol=1e-6)):
        raise ValueError("Expected members x candidates x normalized event probabilities")
    entropy = lambda q: -(q*np.log(np.clip(q, 1e-15, 1.))).sum(-1)
    total = entropy(p.mean(0))
    conditional = entropy(p).mean(0)
    return total, conditional, np.maximum(total-conditional, 0.)


@dataclass(frozen=True)
class JointRiskSettings:
    members: int = 3
    width: int = 64
    embedding: int = 12
    epochs: int = 100
    batch: int = 128
    learning_rate: float = .003
    auxiliary: bool = True
    aggregate: bool = False
    latent: bool = True
    margin_weight: float = .2
    metric_weight: float = .05

    def __post_init__(self):
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1
               for k in ("members", "width", "embedding", "epochs", "batch")):
            raise ValueError("Positive integer network dimensions required")
        if self.members < 2 or any(type(getattr(self, k)) is not bool
                                   for k in ("auxiliary", "aggregate", "latent")):
            raise ValueError("An ensemble and explicit boolean ablations are required")
        if any(not np.isfinite(v) or v < 0 for v in
               (self.learning_rate, self.margin_weight, self.metric_weight)) or self.learning_rate == 0:
            raise ValueError("Invalid learning rates or auxiliary weights")


class EventNetwork(nn.Module):
    def __init__(self, inputs, margins, settings):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, settings.width), nn.SiLU(),
                                     nn.Linear(settings.width, settings.embedding), nn.SiLU())
        self.event = nn.Linear(inputs+settings.embedding, 3 if settings.aggregate else 7)
        self.margin = nn.Linear(settings.embedding, max(1, margins))

    def forward(self, x):
        z = self.encoder(x)
        return self.event(torch.cat((x, z), dim=1)), self.margin(z), z


def weights_digest(models):
    result = hashlib.sha256()
    for i, model in enumerate(models):
        for name, tensor in sorted(model.state_dict().items()):
            result.update((str(i)+":"+name).encode())
            result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class JointRiskEnsemble:
    """Separate risk embedding; the performance VAE stays frozen."""

    def __init__(self, rows, features, seed=0, settings=None):
        validate_history(rows)
        features.assert_frozen()
        if features.history_id != history_identity(rows):
            raise ValueError("Risk model and encoder must have the same training history")
        self.features, self.snapshot_id = features, features.snapshot_id
        self.settings, self.seed = settings or JointRiskSettings(), seed
        self.training_indices = [r["evaluation"] for r in rows]
        self.history_id = history_identity(rows)
        self.training_keys = sorted({digest(features.candidate(r["candidate"]).to_dict()) for r in rows})
        self.events = ("feasible", "explicit_infeasible", "hidden_failure") if self.settings.aggregate else EVENTS
        x = self._inputs([r["candidate"] for r in rows])
        self.center, self.scale = x.mean(0), x.std(0).clamp_min(.05)
        x = (x-self.center)/self.scale
        allowed = torch.tensor(terminal_sets(rows, self.settings.aggregate))
        self.margin_names = sorted({name for r in rows for name in r["risk"]["constraints"]
                                    if name not in r["risk"]["analytical_constraints"]})
        values, mask = np.zeros((len(rows), len(self.margin_names))), np.zeros(
            (len(rows), len(self.margin_names)), dtype=bool)
        for i, row in enumerate(rows):
            for j, name in enumerate(self.margin_names):
                risk = row["risk"]
                if risk["constraint_observed"].get(name, False):
                    values[i, j], mask[i, j] = risk["constraints"][name], True
        means, scales = [], []
        for j in range(values.shape[1]):
            observed = values[mask[:, j], j]
            means.append(float(observed.mean()) if len(observed) else 0.)
            scales.append(max(float(observed.std()), 1e-6) if len(observed) else 1.)
        self.margin_center, self.margin_scale = means, scales
        target = torch.tensor((values-np.asarray(means))/np.asarray(scales), dtype=torch.float32)
        observed = torch.tensor(mask)
        self.models, traces = [], []
        for member in range(self.settings.members):
            with torch.random.fork_rng():
                torch.manual_seed(seed+1009*member)
                generator = torch.Generator().manual_seed(seed+1009*member+77)
                model = EventNetwork(x.shape[1], len(self.margin_names), self.settings)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.settings.learning_rate,
                                              weight_decay=1e-4)
                bootstrap = torch.randint(len(rows), (len(rows),), generator=generator)
                last = None
                for _ in range(self.settings.epochs):
                    order = bootstrap[torch.randperm(len(rows), generator=generator)]
                    for indices in order.split(self.settings.batch):
                        logits, margins, embedding = model(x[indices])
                        loss = partial_event_loss(logits, allowed[indices])
                        if self.settings.auxiliary:
                            if observed[indices].any():
                                difference = margins[:, :len(means)]-target[indices]
                                loss = loss+self.settings.margin_weight*F.smooth_l1_loss(
                                    difference[observed[indices]], torch.zeros_like(difference[observed[indices]]))
                            loss = loss+self.settings.metric_weight*supervised_metric_loss(
                                embedding, allowed[indices])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite joint risk training loss")
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                        optimizer.step()
                        last = float(loss.detach())
                self.models.append(model.eval())
                traces.append({"member": member, "last_batch_loss": last,
                               "bootstrap_indices_sha256": digest(bootstrap.tolist())})
        counts = {name: int(((allowed.sum(1) == 1) & allowed[:, j]).sum())
                  for j, name in enumerate(self.events)}
        self.diagnostics = {
            "event_support": counts, "unsupported_events": [k for k, v in counts.items() if v == 0],
            "partial_event_rows": int((allowed.sum(1) > 1).sum()),
            "unknown_event_rows": int((allowed.sum(1) == 0).sum()),
            "margin_observations": dict(zip(self.margin_names, mask.sum(0).astype(int).tolist())),
            "training_trace": traces, "probability_guarantee": False,
            "uncertainty": "bootstrap entropy difference, not an exact Bayesian posterior"}
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.weight_id = weights_digest(self.models)
        self.model_id = digest(self._metadata())

    def _inputs(self, candidates):
        onehot = self.features.transform_view(candidates, "hierarchical_onehot").float()
        if not self.settings.latent:
            return onehot
        latent = self.features.transform_view(candidates, "latent")[:, -self.features.settings.latent_dim:].float()
        return torch.cat((onehot, latent), 1)

    def assert_frozen(self):
        self.features.assert_frozen()
        if (self.features.snapshot_id != self.snapshot_id or weights_digest(self.models) != self.weight_id
                or digest(self._metadata()) != self.model_id):
            raise RuntimeError("Joint risk ensemble changed outside its snapshot")

    def predict(self, candidates):
        self.assert_frozen()
        with torch.no_grad():
            x = (self._inputs(candidates)-self.center)/self.scale
            members = np.asarray([F.softmax(m(x)[0], 1).numpy() for m in self.models])
        total, conditional, information = entropy_decomposition(members)
        return {"events": list(self.events), "probabilities": members.mean(0).tolist(),
                "members": members.tolist(), "final_feasibility": members.mean(0)[:, 0].tolist(),
                "total_entropy": total.tolist(), "conditional_entropy": conditional.tolist(),
                "information": information.tolist(), "model_id": self.model_id,
                "unsupported_events": self.diagnostics["unsupported_events"]}

    def _metadata(self):
        return {
            "schema": "ra_joint_risk_v1", "settings": asdict(self.settings), "seed": self.seed,
            "training_indices": self.training_indices, "training_keys": self.training_keys,
            "history_id": self.history_id, "snapshot_id": self.snapshot_id, "events": list(self.events),
            "center": self.center.tolist(), "scale": self.scale.tolist(),
            "margin_names": self.margin_names, "margin_center": self.margin_center,
            "margin_scale": self.margin_scale, "diagnostics": self.diagnostics,
            "source_sha256": self.source_sha256, "weight_id": self.weight_id}

    def save_checkpoint(self, path):
        self.assert_frozen()
        with Path(path).open("xb") as stream:
            torch.save({"metadata_json": json.dumps(self._metadata(), sort_keys=True, allow_nan=False),
                        "models": [m.state_dict() for m in self.models], "model_id": self.model_id}, stream)

    @classmethod
    def load_checkpoint(cls, path, features):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = json.loads(payload["metadata_json"])
        if (metadata.get("schema") != "ra_joint_risk_v1" or features.snapshot_id != metadata["snapshot_id"]
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != metadata["source_sha256"]):
            raise ValueError("Risk checkpoint source or feature identity differs")
        result = cls.__new__(cls)
        result.features = features
        for name, value in metadata.items():
            if name not in ("schema", "settings", "center", "scale"):
                setattr(result, name, value)
        result.settings = JointRiskSettings(**metadata["settings"])
        result.center = torch.tensor(metadata["center"], dtype=torch.float32)
        result.scale = torch.tensor(metadata["scale"], dtype=torch.float32)
        result.models = []
        for state in payload["models"]:
            model = EventNetwork(len(result.center), len(result.margin_names), result.settings)
            model.load_state_dict(state, strict=True)
            result.models.append(model.eval())
        result.model_id = payload["model_id"]
        result.assert_frozen()
        return result


class ReferenceTemperature:
    """Repeated fixed-reference calibration, with no exchangeability guarantee."""

    def __init__(self, model, rows):
        validate_history(rows)
        model.assert_frozen()
        indices = [r["evaluation"] for r in rows]
        keys = {digest(model.features.candidate(r["candidate"]).to_dict()) for r in rows}
        if set(indices) & set(model.training_indices) or keys & set(model.training_keys):
            raise ValueError("Calibration reference overlaps encoder/risk training")
        pairs = [(r["candidate"], int(r["risk"]["feasible_label"])) for r in rows
                 if r["risk"]["feasible_label"] is not None]
        self.model_id, self.indices, self.history_id = model.model_id, indices, history_identity(rows)
        self.temperature, self.status = 1., "identity_insufficient_classes"
        self.samples, self.positives = len(pairs), sum(y for _, y in pairs)
        if len(pairs) >= 30 and min(self.positives, len(pairs)-self.positives) >= 5:
            p = model.predict([c for c, _ in pairs])["final_feasibility"]
            scores, labels = logit(np.clip(p, 1e-7, 1.-1e-7)), np.asarray([y for _, y in pairs])

            def objective(log_temperature):
                z = scores/np.exp(log_temperature)
                return np.mean(np.logaddexp(0., z)-labels*z)+.05*log_temperature**2

            fitted = minimize_scalar(objective, bounds=(np.log(.3), np.log(3.)), method="bounded")
            if not fitted.success or not np.isfinite(fitted.fun):
                raise RuntimeError("Temperature fit failed")
            self.temperature, self.status = float(np.exp(fitted.x)), "fitted_reference_temperature"
        self.calibration_id = digest(self.to_dict())

    def transform(self, probabilities, model_id):
        if digest(self.to_dict()) != self.calibration_id:
            raise RuntimeError("Reference calibration changed outside its snapshot")
        if model_id != self.model_id:
            raise ValueError("Calibration belongs to another fitted model")
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1.):
            raise ValueError("Normalized event probabilities required")
        if self.status != "fitted_reference_temperature":
            return p.copy()
        calibrated = expit(logit(np.clip(p[:, 0], 1e-7, 1.-1e-7))/self.temperature)
        result = p.copy()
        result[:, 1:] *= ((1.-calibrated)/np.maximum(1.-p[:, 0], 1e-12))[:, None]
        # At an exact unit event probability, preserve the missing conditional distribution uniformly.
        exact = p[:, 0] >= 1.-1e-12
        result[exact, 1:] = ((1.-calibrated[exact])/(p.shape[1]-1))[:, None]
        result[:, 0] = calibrated
        return result

    def to_dict(self):
        return {"model_id": self.model_id, "indices": self.indices, "history_id": self.history_id,
                "temperature": self.temperature, "status": self.status, "samples": self.samples,
                "positives": self.positives, "guarantee": False,
                "usage": "Fixed reference reused adaptively; not an independent efficacy test"}
