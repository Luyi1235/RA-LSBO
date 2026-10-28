"""Opt-in native BEM/VLM mission coupling with circulation-based induced drag.

Axial slipstream is included in both boundary conditions and pressure velocity.
Induced drag uses a planar lifting-line Fourier approximation, not the installed
half-wing Trefftz routine. No rotor-count reward or energy multiplier is used.
"""

from copy import deepcopy
from functools import wraps
from collections import OrderedDict
import hashlib
import ast
import importlib
import inspect
import math
import numpy as np

VERSION = 'native_mission_increment_20261004_v2'
DEFAULTS = dict(version=VERSION, enabled=True, spanwise_panels=24, chordwise_panels=4)


class InfluenceSolveCache:
    """Exact matrix-keyed LU reuse; every new RHS is solved without interpolation."""
    def __init__(self, limit=32):
        self.limit, self.cache, self.hits, self.misses = limit, OrderedDict(), 0, 0

    def solve(self, a, b):
        from scipy.linalg import lu_factor, lu_solve
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim == 3 and b.ndim == 3 and len(a) == len(b):
            return np.stack([self.solve(x,y) for x,y in zip(a,b)])
        if a.ndim != 2:
            return np.linalg.solve(a,b)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite VLM linear system')
        key=(a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest())
        if key in self.cache:
            self.hits += 1
            factors=self.cache.pop(key)
        else:
            self.misses += 1
            factors=lu_factor(a,check_finite=False)
        self.cache[key]=factors
        while len(self.cache)>self.limit:
            self.cache.popitem(last=False)
        solution=lu_solve(factors,b,check_finite=False)
        if not np.isfinite(solution).all():
            raise ValueError('Singular native VLM linear system')
        return solution


class _NamespaceProxy:
    def __init__(self, original, **overrides):
        self._original=original
        self.__dict__.update(overrides)

    def __getattr__(self, key):
        return getattr(self._original,key)


class InfluenceGeometryCache:
    """Reuse native influence coefficients only for byte-identical consumed inputs."""
    def __init__(self, native):
        self.native, self.key, self.value = native, None, None
        self.hits, self.misses = 0, 0
        tree = ast.parse(inspect.getsource(native))
        self.fields = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == 'VD'})
        if not self.fields:
            raise RuntimeError('Unrecognized native influence geometry interface')

    @staticmethod
    def fingerprint(value):
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError('Unsupported object-valued native influence input')
        return (a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest())

    def __call__(self, VD, mach, compute_EW=False):
        key = (bool(compute_EW), self.fingerprint(mach),
               tuple((k,self.fingerprint(getattr(VD,k))) for k in self.fields))
        if key == self.key:
            self.hits += 1
            return self.value
        self.misses += 1
        value = self.native(VD,mach,compute_EW=compute_EW)
        for item in value:
            if isinstance(item,np.ndarray):
                item.setflags(write=False)
        self.key,self.value = key,value
        return value


def normalize(settings):
    if (not isinstance(settings, dict) or set(settings) != set(DEFAULTS)
            or settings['version'] != VERSION or type(settings['enabled']) is not bool):
        raise ValueError('Explicit versioned mission-slipstream settings required')
    for key, lo, hi in (('spanwise_panels', 12, 64), ('chordwise_panels', 2, 8)):
        if type(settings[key]) is not int or not lo <= settings[key] <= hi:
            raise ValueError('Invalid mission-slipstream panel count')
    return dict(settings)


def add_axial_pressure_velocity(rhs):
    """Kutta-Joukowski loading must use local axial speed, not just freestream."""
    velocity = np.asarray(rhs.V_distribution, dtype=float)
    axial = np.asarray(rhs.Vx_ind_total, dtype=float)
    if (velocity.shape != axial.shape or np.any(velocity <= 0)
            or not np.isfinite(velocity).all() or not np.isfinite(axial).all()):
        raise ValueError('Invalid native axial pressure velocity')
    rhs.ONSET = np.asarray(rhs.ONSET) + axial / velocity
    return rhs


def fourier_drag(y, gamma_over_v, span, area, modes=12):
    """Gamma/V = 2*b*sum(A_n sin(n*theta)); CDi=pi*AR*sum(n*A_n**2).

    Fit the complete wing, not two independent root-truncated semi-wings.
    The model neglects nonplanar-wake and inter-wing induced interference.
    """
    y, g = np.asarray(y, dtype=float), np.atleast_2d(gamma_over_v).astype(float)
    if (not math.isfinite(span) or not math.isfinite(area) or min(span, area) <= 0
            or y.ndim != 1 or g.shape[1] != len(y) or len(y) < 6
            or not np.isfinite(y).all() or not np.isfinite(g).all()
            or np.any(np.abs(y) >= span/2 + 1e-6)):
        raise ValueError('Invalid projected lifting-line data')
    theta = np.arccos(np.clip(2*y/span, -1., 1.))
    orders = np.arange(1, min(modes, len(y)//2)+1)
    basis = np.sin(theta[:, None]*orders)
    coefficients, _, rank, _ = np.linalg.lstsq(basis, (g/(2*span)).T, rcond=1e-10)
    if rank != len(orders):
        raise ValueError('Rank-deficient full-wing circulation fit')
    cdi = np.pi*span**2/area * np.sum(orders[:, None]*coefficients**2, axis=0)
    return cdi[:, None]


def circulation_drag(result, vehicle):
    vd = result.VD
    gamma = np.add.reduceat(result.gamma, np.asarray(vd.chordwise_breaks, dtype=int), axis=1)
    y = np.asarray(vd.YCH)[np.asarray(vd.leading_edge_indices, dtype=bool)]
    from RCAIDE.Framework.Core import Data
    by_wing, total = Data(), np.zeros((len(gamma), 1))
    surface, strip = 0, 0
    for wing in vehicle.wings:
        count = 2 if wing.symmetric else 1
        n = int(np.sum(vd.n_sw[surface:surface+count]))
        # Missions are symmetric, zero-sideslip and do not trim lateral moments.
        if wing.vertical:
            drag = np.zeros_like(total)
        else:
            center = float(wing.origin[0][1])
            drag = fourier_drag(y[strip:strip+n]-center, gamma[:, strip:strip+n],
                                float(wing.spans.projected), float(wing.areas.reference))
        by_wing[wing.tag] = drag
        total += drag * float(wing.areas.reference)/float(vehicle.reference_area)
        strip += n
        surface += count
    if strip != gamma.shape[1]:
        raise ValueError('Unsupported extra lifting surfaces in circulation model')
    return total, by_wing


def install_native_pressure_patch():
    module = importlib.import_module('RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.VLM')
    original = module.compute_RHS_matrix
    if getattr(original, '_lsbo_pressure_velocity', False):
        raise ValueError('Native pressure velocity patch already installed')
    if 'FACTOR = FORAXL + ONSET' not in inspect.getsource(module.VLM):
        raise RuntimeError('Native VLM force formulation changed; review pressure coupling')
    @wraps(original)
    def rhs(*args, **kwargs):
        return add_axial_pressure_velocity(original(*args, **kwargs))
    rhs._lsbo_pressure_velocity = True
    module.compute_RHS_matrix = rhs
    cache = InfluenceSolveCache()
    module.np = _NamespaceProxy(module.np, linalg=_NamespaceProxy(module.np.linalg, solve=cache.solve))
    cache.geometry = InfluenceGeometryCache(module.compute_wing_induced_velocity)
    module.compute_wing_induced_velocity = cache.geometry
    return cache


def evaluate_lift(state, settings, vehicle):
    from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.VLM import VLM
    from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.evaluate_VLM import evaluate_surrogate
    c = state.conditions
    if np.any(np.abs(c.aerodynamics.angles.beta) > 1e-8):
        raise ValueError('Coupled conceptual model supports symmetric mission flight only')
    # Preserve the dry reference. Only a matched native wake-on/off increment is added.
    evaluate_surrogate(state, settings, vehicle)
    aero = c.aerodynamics.coefficients
    configured = settings.mission_slipstream
    delta_lift, delta_drag = np.zeros_like(aero.lift.total), np.zeros_like(aero.lift.total)
    wake_speed = np.zeros_like(delta_lift)
    if configured['enabled']:
        options = deepcopy(settings)
        options.use_surrogate = False
        options.number_of_spanwise_vortices = configured['spanwise_panels']
        options.number_of_chordwise_vortices = configured['chordwise_panels']
        options.propeller_wake_model = False
        off = VLM(c, options, vehicle)
        drag_off, wings_off = circulation_drag(off, vehicle)
        options.propeller_wake_model = True
        on = VLM(c, options, vehicle)
        drag_on, wings_on = circulation_drag(on, vehicle)
        delta_lift = on.CLift-off.CLift
        delta_drag = drag_on-drag_off
        if not np.isfinite(delta_lift).all() or not np.isfinite(delta_drag).all():
            raise ValueError('Nonfinite native mission aerodynamic increment')
        aero.lift.total = aero.lift.total + delta_lift
        aero.drag.induced.inviscid = aero.drag.induced.inviscid + delta_drag
        for wing in vehicle.wings:
            tag = wing.tag
            dcl = on.CLift_wings[tag]-off.CLift_wings[tag]
            # Assign, rather than mutate: RCAIDE may alias these two dictionaries.
            updated_lift = aero.lift.induced.inviscid_wings[tag] + dcl
            aero.lift.induced.inviscid_wings[tag] = updated_lift
            aero.lift.compressible_wings[tag] = updated_lift
            aero.drag.induced.inviscid_wings[tag] = aero.drag.induced.inviscid_wings[tag] + wings_on[tag]-wings_off[tag]
        for name, field in (('Y','CY'),('L','CL'),('M','CM'),('N','CN')):
            c.static_stability.coefficients[name] = c.static_stability.coefficients[name] + on[field]-off[field]
        wake_speed = np.max(on.V_x, axis=1)[:,None]
    c.static_stability.coefficients.lift = aero.lift.total
    c.static_stability.coefficients.drag = aero.drag.induced.inviscid
    c.aerodynamics.mission_axial_wake_max_mps = wake_speed
    c.aerodynamics.mission_circulation_drag_increment = delta_drag
    c.aerodynamics.mission_native_lift_increment = delta_lift


def install(nexus, diagnostics, config):
    settings = normalize(config['mission_slipstream'])
    research = config.get('propulsion_research', {})
    if (research.get('native_wake', True) or not research.get('synchronize_mission_mass')
            or not research.get('correct_nacelle_drag') or 'section_consistency' not in config
            or any(k in config for k in ('dep_groups','powered_lift_probe'))):
        raise ValueError('Mission coupling requires consistent physics, no legacy wake/group/probe mode')
    solve_cache = install_native_pressure_patch()
    from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake import wake_model
    from .propulsion_research import rowwise_axial_wake, statistics
    original_wake = wake_model.compute_wake_induced_velocity
    def axial(rotor, conditions, points, count):
        return rowwise_axial_wake(original_wake, rotor, conditions, points, count)
    wake_model.compute_wake_induced_velocity = axial
    report = diagnostics['mission_slipstream'] = dict(settings=settings,
        native_pressure_velocity_corrected=True,
        induced_drag='paired_full_wing_Fourier_increment_on_unchanged_dry_model',
        mission_energy='original_battery_integral_no_discount',
        limitations=['axial_wake_no_swirl', 'planar_far_wake_no_interwing_interference',
                     'no_viscous_separation', 'fixed_nonwing_mass', 'all_rotors_active'],
        status='not_run')
    mission = nexus.missions.base_mission
    for segment in mission.segments.values():
        solver = segment.state.numerics.solver
        if solver.objective is not None:
            raise ValueError('Coupled residual solver cannot replace a mission objective')
        solver.lsbo_domain_first = True
        solver.max_evaluations = min(int(solver.max_evaluations), 200)
    report['mission_solver'] = 'existing_domain_backtracking_used_as_primary_residual_solver'
    original_aero = mission.process.initialize.aero
    @wraps(original_aero)
    def initialize(m):
        for seg in m.segments.values():
            a = seg.analyses.aerodynamics
            a.settings.use_surrogate = True
            a.settings.propeller_wake_model = False
            a.settings.mission_slipstream = deepcopy(settings)
        original_aero(m)
        for seg in m.segments.values():
            seg.analyses.aerodynamics.process.compute.lift.inviscid_wings = evaluate_lift
    mission.process.initialize.aero = initialize
    native_post = nexus.procedure.post_process
    @wraps(native_post)
    def post(candidate):
        result = native_post(candidate)
        rows = []
        for seg in candidate.results.base.segments.values():
            aero = seg.conditions.aerodynamics
            rows.append(dict(segment=seg.tag,
                axial_wake_mps=statistics(aero.mission_axial_wake_max_mps),
                circulation_drag_increment=statistics(aero.mission_circulation_drag_increment),
                native_lift_increment=statistics(aero.mission_native_lift_increment),
                alpha_deg=statistics(np.rad2deg(aero.angles.alpha))))
        report.update(status='evaluated', segments=rows)
        report['linear_solve_cache'] = dict(hits=solve_cache.hits, misses=solve_cache.misses,
            entries=len(solve_cache.cache), policy='exact_matrix_LU_reuse_no_interpolation')
        report['influence_geometry_cache'] = dict(hits=solve_cache.geometry.hits,
            misses=solve_cache.geometry.misses, fields=solve_cache.geometry.fields,
            policy='exact_consumed_geometry_and_Mach_no_interpolation')
        return result
    nexus.procedure.post_process = post
