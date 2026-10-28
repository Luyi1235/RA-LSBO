"""Explicit feature-view adapters for risk heads and the residual GP interface."""

from dataclasses import asdict, dataclass

import torch

from .budget import digest
from .calibration import CalibratedRiskHeads
from .features import history_identity, validate_history
from .responses import ResponseHeads
from .risk_model import RiskHeads
from .structured_encoder import EncoderSettings, StructuredFeatures, VIEWS
from .training import training_rows


@dataclass(frozen=True)
class EncoderFeatureView:
    encoder: StructuredFeatures
    view_name: str

    def __post_init__(self):
        if self.view_name not in VIEWS:
            raise ValueError("Unknown encoder feature view")

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def transform(self, candidates):
        return self.encoder.transform_view(candidates, self.view_name)

    def layout(self):
        return self.encoder.layout(self.view_name)


class StructuredRiskHeads(RiskHeads):
    """Model identity binds both the learned encoder and the chosen risk input."""

    def __init__(self, rows, features, seed=0, view="latent", **kwargs):
        if not isinstance(features, StructuredFeatures):
            raise ValueError("Structured encoder required")
        super().__init__(rows, EncoderFeatureView(features, view), seed, **kwargs)
        self.model_id = digest({
            "base_risk_model_id": self.model_id, "feature_layout": self.features.layout()})
        for diagnostic in self.diagnostics.values():
            diagnostic["feature_view"] = view
            diagnostic["input_dimensions"] = self.features.layout()["dimensions"]


class StructuredResponseHeads(ResponseHeads):
    """Reuse the existing GP fit/predict code with a declared joint-input layout."""

    def __init__(self, rows, features, seed=0, steps=45, use_prior=True):
        validate_history(rows)
        features.assert_frozen()
        if (type(steps) is not int or steps < 1
                or features.training_indices != [r["evaluation"] for r in rows]
                or features.history_id != history_identity(rows)):
            raise ValueError("Response heads require the same training prefix and positive steps")
        self.features, self.snapshot_id = features, features.snapshot_id
        self.models, self.diagnostics = {}, {}
        rep = features.representation
        with torch.no_grad():
            latent, _ = rep.model.encode(rep.grammar)
        base_dimensions = features.layout()["dimensions"]-features.settings.latent_dim
        self.offset = torch.cat([torch.zeros(base_dimensions), latent.min(0).values-.5]).double()
        upper = torch.cat([torch.ones(base_dimensions), latent.max(0).values+.5]).double()
        self.scale = (upper-self.offset).clamp_min(1e-8)
        tables = training_rows(rows)
        datasets = {"objective": tables["objective"],
                    **{"constraint:"+name: table for name, table in tables["constraints"].items()}}
        for name, table in datasets.items():
            diagnostic = {
                "samples": len(table), "training_indices": [r["evaluation"] for r in table],
                "representation_snapshot": self.snapshot_id, "calibrated": False,
                "feature_layout": features.layout()}
            self.diagnostics[name] = diagnostic
            if len(table) < 3:
                diagnostic["status"] = "unavailable_insufficient_observations"
                continue
            x = (features.transform([r["candidate"] for r in table])-self.offset)/self.scale
            sign = -1. if name == "objective" else 1.
            values = sign*torch.tensor([r["target"] for r in table], dtype=torch.double)
            center, scale = float(values.mean()), float(values.std().clamp_min(1e-6))
            target = ((values-center)/scale).unsqueeze(-1)
            base = features.modules["surrogate"]
            prior = (base.FrozenPerformanceMean(rep, center, scale, self.offset, self.scale)
                     if name == "objective" and use_prior and rep.performance_fitted else None)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = base.fit_gp(x, target, steps, mean_module=prior, name=name)
            self.models[name] = model, center, scale, sign
            diagnostic.update(status="fitted_raw", prior_used=prior is not None,
                              prior_input="latent_only" if prior is not None else None,
                              fit=model.fit_diagnostics)


def fit_structured_models(base_project, variables, training, calibration=None,
                          encoder_settings=None, encoder_seed=0, risk_seed=0,
                          risk_view="latent", gp_steps=45):
    """One chronological block; no implicit future data or previous calibrator."""
    features = StructuredFeatures(base_project, variables, encoder_seed, encoder_settings).fit(training)
    risks = StructuredRiskHeads(training, features, risk_seed, view=risk_view)
    responses = StructuredResponseHeads(training, features, risk_seed, steps=gp_steps)
    calibrated = None if calibration is None else CalibratedRiskHeads(risks, calibration)
    return features, risks, responses, calibrated


class StructuredLearningStrategy:
    """Opt-in adapter for the existing loop; the legacy encoder stays the default."""

    def __init__(self, encoder_settings=None, risk_view="latent", incremental=True):
        self.settings = encoder_settings or EncoderSettings()
        if risk_view not in VIEWS or type(incremental) is not bool:
            raise ValueError("Explicit risk view and boolean update policy required")
        self.risk_view, self.incremental, self.features = risk_view, incremental, None

    def should_refresh(self, rows):
        return False

    def fit(self, rows, variables, base_project, settings):
        boundary = len(rows)-settings.calibration_window
        training, calibration = rows[:boundary], rows[boundary:]
        if len(training) < 3 or not calibration:
            raise ValueError("Nonempty chronological calibration and training >=3 required")
        update = "reset"
        if self.features is None or not self.incremental:
            self.features = StructuredFeatures(base_project, variables, settings.seed, self.settings).fit(training)
        else:
            self.features.update(training)
            update = "append_only_incremental"
        risks = StructuredRiskHeads(training, self.features, settings.seed, self.risk_view)
        responses = StructuredResponseHeads(training, self.features, settings.seed, settings.gp_steps)
        calibrated = CalibratedRiskHeads(risks, calibration)
        return (risks, responses, calibrated), {
            "model_policy": "structured_encoder_v1", "encoder_update": update,
            "encoder_settings": asdict(self.settings),
            "risk_feature_layout": risks.features.layout(),
            "response_feature_layout": self.features.layout(),
            "encoder_objective_indices": self.features.objective_indices,
            "encoder_training_trace": self.features.representation.training_trace}
