"""One-step Brier reduction from the observable signs of three aircraft margins."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import ndtr

LOOKAHEAD_HEADS = ("stall", "root_bending", "soc")
NODES, WEIGHTS = np.polynomial.legendre.leggauss(24)


def posterior_cross_covariance(model, left, right):
    """Cross-covariance under the fitted Gaussian EP approximation."""
    cross = model._baseline_covariance(left, right)
    if len(model.interval_x):
        left_cross = model._baseline_covariance(left, model.interval_x)
        right_cross = model._baseline_covariance(model.interval_x, right)
        left_projection = cho_solve(model._interval_factor, left_cross.T).T
        right_projection = cho_solve(model._interval_factor, right_cross)
        cross -= left_projection @ right_cross
        cross += left_projection @ model._posterior_covariance @ right_projection
    return cross


def sign_covariance(query_z, reference_z, correlation):
    """Covariance of two Gaussian sign indicators.

    Integrate the derivative of the bivariate normal CDF from zero correlation
    to the given correlation, with rho=sin(theta) to remove the endpoint factor.
    """
    q, r = query_z[:, None], reference_z[None, :]
    theta = np.arcsin(np.clip(correlation, -1., 1.))
    integral = np.zeros_like(theta)
    for node, weight in zip(NODES, WEIGHTS):
        angle = theta * ((1. + node) / 2.)
        rho = np.sin(angle)
        exponent = -0.25 * ((q-r)**2 / (1-rho) + (q+r)**2 / (1+rho))
        integral += weight * np.exp(np.minimum(exponent, 0.))
    return theta * integral / (4*np.pi)


def sign_variance_reduction(query_z, reference_z, correlation):
    """Var(E[1{g(y)<=0} | 1{g(x)<=0}]) for a bivariate Gaussian."""
    covariance = sign_covariance(query_z, reference_z, correlation)
    q, r = query_z[:, None], reference_z[None, :]
    p_query, p_reference = ndtr(q), ndtr(r)
    denominator = p_query * ndtr(-q)
    reduction = np.divide(covariance**2, denominator, out=np.zeros_like(covariance),
                          where=denominator > 1e-14)
    return np.clip(reduction, 0., p_reference * ndtr(-r))


def brier_scores(heads, query, reference):
    """Mean reduction over an unlabeled reference panel, with fixed GP sites.

    The acquisition forecasts sign-only feedback for three heads and holds the
    other heads fixed. Real updates use all supported numeric/interval outputs.
    """
    current_joint = np.ones(len(reference))
    future_second_moment = np.ones((len(query), len(reference)))
    for name, head in heads.items():
        model = head.model
        ref = reference / head.divisor
        ref_moments = model.predict(ref)
        threshold = -head.center/head.scale
        ref_sd = np.sqrt(np.maximum(ref_moments["variance"], 1e-14))
        ref_z = (threshold-ref_moments["mean"])/ref_sd
        p = ndtr(ref_z)
        current_joint *= p
        second_moment = p**2
        if name in LOOKAHEAD_HEADS:
            proposed = query / head.divisor
            moments = model.predict(proposed)
            sd = np.sqrt(np.maximum(moments["variance"], 1e-14))
            z = (threshold-moments["mean"])/sd
            covariance = posterior_cross_covariance(model, proposed, ref)
            correlation = covariance / sd[:, None] / ref_sd[None, :]
            if not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1.00001):
                raise FloatingPointError("Invalid EP posterior correlation for aircraft sign lookahead")
            second_moment = second_moment + sign_variance_reduction(z, ref_z, correlation)
        future_second_moment *= second_moment
    return np.maximum(future_second_moment-current_joint**2, 0.).mean(axis=1)
