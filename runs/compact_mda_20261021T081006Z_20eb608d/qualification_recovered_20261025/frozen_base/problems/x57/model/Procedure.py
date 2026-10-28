# Procedure.py — v10
# 机翼重量: Torenbeek Appendix C 详细公式 (直接迭代求解)
# 关键改进:
#   - 展长直接建模 b^1.675 (非 AR^0.6 × S^0.758)
#   - 根梢比惩罚 k_λ = (1+λ)^0.4 (非 λ^0.04≈无)
#   - t/c指数 -0.45 (非 -0.3)
#   - 含 k_no/k_uc/k_st 修正因子
#   - 每个设计点独立迭代求解翼重-MTOW耦合

import RCAIDE
from RCAIDE.Framework.Core import Units, Data
from RCAIDE.Framework.Analyses.Process import Process
from RCAIDE.Library.Methods.Geometry.Planform import wing_segmented_planform
import numpy as np
import math

# ------------------------------------------------------------------
# X-57 Mod II 基准参数
# ------------------------------------------------------------------
_REF = {
    'b':     11.4,     # m, 基准展长
    'tc':    0.12,     # 相对厚度
    'lambda': 0.630,   # tip/root
    'MTOW':  1325.5,   # kg (标定后基准)
    'W_wing': 169.5,   # kg (标定后基准翼重)
    'n_ult': 3.75,     # 极限载荷
}

_FIXED_MASS = _REF['MTOW'] - _REF['W_wing']  # ≈ 1156 kg

# Torenbeek公式标定常数 (通过基准点迭代求解确定)
_TORENBEEK_C = None


def _torenbeek_raw(b, tc, taper_ratio, sweep_deg, mtow, w_wing, n_ult=3.75):
    """Torenbeek Appendix C 翼重公式 (不含标定常数)

    W_wing = C × k_no × k_λ × k_e × k_uc × k_st
           × [n_ult × (0.8W_s + 0.2W_to)]^0.55
           × b^1.675 × (t/c)^(-0.45) × (cosΛ)^(-1.325)

    参数:
      b:            展长 (m)
      tc:           相对厚度
      taper_ratio:  tip/root taper ratio (0<λ<=1)
      sweep_deg:    1/4弦后掠角 (度)
      mtow:         MTOW (kg)
      w_wing:       当前翼重估计 (kg), 用于计算悬吊重量
      n_ult:        极限载荷因子
    """
    b_s = b / math.cos(math.radians(sweep_deg))
    lambda_tip_root = min(max(taper_ratio, 1.0e-3), 1.0)

    k_no = 1.0 + math.sqrt(1.905 / max(b_s, 1.0))
    k_lambda = (1.0 + lambda_tip_root) ** 0.4
    k_e = 1.0
    k_uc = 0.95

    if mtow > 0:
        v_ne = 80.0
        flutter_term = (b_s**3 / mtow) * (v_ne / 100.0 / max(tc, 0.01))**2 * math.cos(math.radians(sweep_deg))
        k_st = 1.0 + 9.06e-4 * flutter_term
    else:
        k_st = 1.0

    w_suspended = mtow - w_wing
    if w_suspended < 500:
        w_suspended = 500.0
    load_mass = 0.8 * w_suspended + 0.2 * mtow
    load_factor = (n_ult * load_mass) ** 0.55

    geom = b_s**1.675 * tc**(-0.45) * math.cos(math.radians(sweep_deg))**(-1.325)

    return k_no * k_lambda * k_e * k_uc * k_st * load_factor * geom


def _calibrate_torenbeek():
    """在基准点迭代求解标定常数 C"""
    global _TORENBEEK_C
    if _TORENBEEK_C is not None:
        return _TORENBEEK_C

    b, tc, lam = _REF['b'], _REF['tc'], _REF['lambda']
    target = _REF['W_wing']

    w = target  # initial guess
    for _ in range(20):
        mtow = _FIXED_MASS + w
        raw = _torenbeek_raw(b, tc, lam, 0.0, mtow, w, _REF['n_ult'])
        # C = target / raw, but w affects raw through load term
        # Solve: C × raw(C) = target, MTOW = FIXED_MASS + target
        # Actually we just compute C once converged
        w_new = w  # keep same w for convergence check
        if abs(mtow - _REF['MTOW']) < 0.01:
            break

    # At converged reference: MTOW = FIXED_MASS + target
    raw_ref = _torenbeek_raw(b, tc, lam, 0.0, _REF['MTOW'], target, _REF['n_ult'])
    _TORENBEEK_C = target / raw_ref
    return _TORENBEEK_C


def _compute_wing_mass_torenbeek(wing):
    """直接迭代求解当前设计的 Torenbeek 翼重

    翼重影响MTOW→MTOW影响载荷项→载荷项影响翼重
    需要内部迭代到收敛

    几何展长用于翼重(结构梁长度), 上反角增加结构长度惩罚。
    """
    # 几何展长 (非投影): sqrt(AR × S), 结构梁的实际长度
    AR = getattr(wing, '_AR_correct', getattr(wing, 'aspect_ratio', 8.8))
    S = getattr(wing.areas, 'reference', 14.76)
    b_geom = np.sqrt(AR * S) if AR > 0 and S > 0 else getattr(wing.spans, 'geometric', 11.4)
    if b_geom <= 0:
        b_geom = getattr(wing.spans, 'projected', 11.4)

    tc = getattr(wing, '_tc_correct', getattr(wing, 'thickness_to_chord', 0.12))
    taper = getattr(wing, '_taper_correct', getattr(wing, 'taper', 0.630))
    sweep_obj = getattr(wing, '_sweep_correct', getattr(wing.sweeps, 'quarter_chord', 0.0))
    if hasattr(sweep_obj, 'degrees'):
        sweep_deg = sweep_obj / Units.deg
    else:
        sweep_deg = float(sweep_obj)

    # ── 上反角结构惩罚 ──
    dihedral_deg = 0.0
    for seg in wing.segments:
        if hasattr(seg, 'dihedral_outboard'):
            dh = seg.dihedral_outboard
            if hasattr(dh, 'degrees'):
                dihedral_deg = float(dh / Units.deg)
            else:
                dihedral_deg = float(dh)
            break
    K_dihedral = 1.0 / np.cos(np.radians(dihedral_deg))

    if b_geom <= 0 or tc <= 0 or taper <= 0:
        return _REF['W_wing']

    C = _calibrate_torenbeek()
    w = _REF['W_wing']  # 初值

    for _ in range(15):
        mtow = _FIXED_MASS + w
        raw = _torenbeek_raw(b_geom, tc, taper, sweep_deg, mtow, w, _REF['n_ult'])
        w_new = C * raw * K_dihedral
        if abs(w_new - w) < 0.05:
            return w_new
        w = w_new

    return w


def _apply_main_wing_segment_taper(wing, taper):
    """Update segment chord ratios so the segmented aerodynamic planform matches the scalar taper."""
    taper = float(np.clip(taper, 0.2, 1.0))
    for seg in getattr(wing, 'segments', []):
        if not hasattr(seg, 'tag') or not hasattr(seg, 'root_chord_percent'):
            continue
        tag = str(seg.tag).lower()
        if tag in ('inboard', 'outboard', 'tip', 'winglet'):
            eta = float(getattr(seg, 'percent_span_location', 0.0))
            chord_ratio = 1.0 - (1.0 - taper) * eta
            if tag == 'winglet':
                chord_ratio *= 0.98
            seg.root_chord_percent = max(0.08, float(chord_ratio))


def setup():
    procedure = Process()
    procedure.simple_sizing = simple_sizing
    procedure.weights = weight
    procedure.missions = Process()
    procedure.missions.design_mission = design_mission
    procedure.post_process = post_process
    return procedure


def simple_sizing(nexus):
    configs = nexus.vehicle_configurations
    base = configs.base
    wing = base.wings.main_wing

    if wing.areas.reference > 0 and wing.aspect_ratio > 0:
        b = np.sqrt(wing.aspect_ratio * wing.areas.reference)
    else:
        b = 0.0

    # ── 上反角投影展长 ──
    dihedral_deg = 0.0
    for seg in wing.segments:
        if hasattr(seg, 'dihedral_outboard'):
            dh = seg.dihedral_outboard
            if hasattr(dh, 'degrees'):
                dihedral_deg = float(dh / Units.deg)
            else:
                dihedral_deg = float(dh)
            break
    b_proj = b * np.cos(np.radians(dihedral_deg))
    mac = wing.areas.reference / b_proj if b_proj > 0 else 0.0

    wing.spans.projected = b_proj
    wing.spans.geometric = b  # 几何展长, 供Torenbeek用
    wing.chords.mean_aerodynamic = mac

    taper = getattr(wing, 'taper', 0.630)
    if mac > 0 and taper > 0:
        avg_chord = wing.areas.reference / b_proj
        cr = 2 * avg_chord / (1 + taper)
        ct = taper * cr
        wing.chords.root = cr
        wing.chords.tip = ct
        wing.taper = taper
        _apply_main_wing_segment_taper(wing, taper)
    else:
        cr, ct = wing.chords.root, wing.chords.tip

    wing_origin = wing.origin[0]
    cg_x = wing_origin[0] + 0.25 * mac
    cg_z = wing_origin[2] - 0.2 * mac
    base.mass_properties.center_of_gravity = [[cg_x, 0.0, cg_z]]

    horiz_tail = None
    for w in base.wings:
        if 'horizontal' in w.tag.lower():
            horiz_tail = w
            break
    if horiz_tail is not None and wing.areas.reference > 0:
        AR_ht_original = 3.3**2 / 2.54
        new_S_ht = (2.54 / 14.76) * wing.areas.reference
        horiz_tail.areas.reference = new_S_ht
        new_b_ht = np.sqrt(AR_ht_original * new_S_ht)
        horiz_tail.spans.projected = new_b_ht
        horiz_tail.chords.mean_aerodynamic = new_S_ht / new_b_ht if new_b_ht > 0 else 0.0
        if getattr(horiz_tail, 'taper', 1.0) == 1.0:
            horiz_tail.chords.root = horiz_tail.chords.mean_aerodynamic
            horiz_tail.chords.tip = horiz_tail.chords.mean_aerodynamic

    for config in configs:
        config_w = config.wings.main_wing
        config_w.aspect_ratio = wing.aspect_ratio
        config_w.areas.reference = wing.areas.reference
        config_w.spans.projected = wing.spans.projected
        config_w.spans.geometric = b
        config_w.chords.mean_aerodynamic = wing.chords.mean_aerodynamic
        config_w.chords.root = wing.chords.root
        config_w.chords.tip = wing.chords.tip
        config_w.taper = wing.taper
        _apply_main_wing_segment_taper(config_w, wing.taper)
        config_w.sweeps.quarter_chord = wing.sweeps.quarter_chord
        config_w.thickness_to_chord = wing.thickness_to_chord
        wing_segmented_planform(config_w)
        config.mass_properties.center_of_gravity = base.mass_properties.center_of_gravity
        if horiz_tail is not None:
            ht_cfg = next((w for w in config.wings if 'horizontal' in w.tag.lower()), None)
            if ht_cfg:
                ht_cfg.areas.reference = horiz_tail.areas.reference
                ht_cfg.spans.projected = horiz_tail.spans.projected
                ht_cfg.chords.mean_aerodynamic = horiz_tail.chords.mean_aerodynamic

    return nexus


def weight(nexus):
    vehicle = nexus.vehicle_configurations.base
    weights_analysis = nexus.analyses.base.weights
    weights_analysis.vehicle = vehicle

    output = weights_analysis.evaluate()

    wing = vehicle.wings.main_wing
    rcaide_wing_mass = getattr(getattr(wing, 'mass_properties', None), 'mass', 0.0)

    # ----- Torenbeek翼重直接迭代求解 -----
    new_wing_mass = _compute_wing_mass_torenbeek(wing)
    new_mtow = _FIXED_MASS + new_wing_mass

    wing.mass_properties.mass = new_wing_mass
    vehicle.mass_properties.max_takeoff = new_mtow

    old_cg = np.array(vehicle.mass_properties.center_of_gravity[0])
    wing_cg = np.array(wing.origin[0])
    new_cg = (_FIXED_MASS * old_cg + new_wing_mass * wing_cg) / new_mtow
    vehicle.mass_properties.center_of_gravity = [new_cg.tolist()]

    for config in nexus.vehicle_configurations:
        config.mass_properties.max_takeoff = new_mtow
        config.mass_properties.center_of_gravity = vehicle.mass_properties.center_of_gravity
        if config.tag != 'base':
            config.wings.main_wing.mass_properties.mass = new_wing_mass

    b = getattr(wing.spans, 'projected', 11.4)
    print(f"    [WEIGHT] b={b:.1f}m, RCAIDE {rcaide_wing_mass:.1f} → Torenbeek {new_wing_mass:.1f} kg, "
          f"MTOW {new_mtow:.1f} kg")
    return nexus


def design_mission(nexus):
    mission = nexus.missions.base_mission
    results = mission.evaluate()
    nexus.results = Data()
    nexus.results.base = results
    return nexus


def post_process(nexus):
    results = nexus.results.base
    summary = nexus.summary
    nexus.total_number_of_iterations += 1

    try:
        base_config = nexus.vehicle_configurations.base
        electric_net = base_config.networks.electric
        bus = electric_net.busses[0]
        battery_modules = bus.battery_modules
        num_modules = len(battery_modules)
        module_max_energy_J = float(battery_modules[0].maximum_energy)
        total_capacity_J = num_modules * module_max_energy_J
    except Exception as e:
        print(f"Warning: failed to read battery capacity from vehicle: {e}")
        total_capacity_J = 47000 * 3600

    final_soc = 1.0
    segments = list(results.segments.values())
    if segments:
        for seg in reversed(segments):
            try:
                bats = seg.conditions.energy.bus.battery_modules
                soc_values = []
                for bat in bats:
                    soc = bat.state_of_charge[-1, 0]
                    soc_values.append(soc)
                if soc_values:
                    final_soc = min(soc_values)
                    break
            except (AttributeError, KeyError, IndexError):
                continue

    total_energy_J = total_capacity_J * (1.0 - final_soc)
    summary.total_energy_consumed = total_energy_J / 3600.0
    summary.min_soc_margin = final_soc

    all_converged = True
    for seg in segments:
        if hasattr(seg, 'converged') and not seg.converged:
            all_converged = False
            break
    summary.converged = all_converged

    base_config = nexus.vehicle_configurations.base
    mtow = getattr(base_config.mass_properties, 'max_takeoff', 0.0)
    wing_area = base_config.wings.main_wing.areas.reference
    ar = base_config.wings.main_wing.aspect_ratio
    rho_sl = 1.225
    # ── v13: CL_max依赖t/c (NACA 4-digit实验数据拟合) ──
    tc = getattr(base_config.wings.main_wing, 'thickness_to_chord', 0.12)
    cl_max_2d = 2.0 * (tc / 0.12) ** 0.15
    e_oswald = 0.8
    cl_max_3d = cl_max_2d / (1.0 + cl_max_2d / (np.pi * ar * e_oswald))
    if wing_area > 0 and mtow > 0:
        v_stall = np.sqrt(2.0 * mtow * 9.81 / (rho_sl * wing_area * cl_max_3d))
        wing_loading = mtow * 9.81 / wing_area
    else:
        v_stall = 0.0
        wing_loading = 0.0
    summary.stall_speed = v_stall
    summary.wing_loading = wing_loading
    summary.cl_max_3d = cl_max_3d

    return nexus
