"""Offline prerequisite study; no objective rewards, constraint credit or BO data."""

import hashlib
import math
from pathlib import Path
import re
from .dep_physics import footprint, lift_requirement, conditional_power_requirement, VERSION
from .propulsion_geometry import make_layout
from .model.design_space import CONTINUOUS_SPECS


def polar_inventory(folder, airfoil, required_reynolds):
    """Report data coverage, never extrapolate or certify an XFOIL CLmax."""
    prefix = 'Clark_y' if airfoil == 'Clark_Y' else airfoil
    records = []
    for path in sorted(Path(folder).glob(prefix+'_polar_Re_*.txt')):
        raw = path.read_bytes()
        text = raw.decode('utf-8',errors='replace')
        match = re.search(r'Re\s*=\s*([\d.]+)\s*e\s*([+-]?\d+)',text)
        if not match:
            raise ValueError('Missing Reynolds metadata in '+str(path))
        rows = []
        table = False
        for line in text.splitlines():
            if '------' in line:
                table = True
                continue
            fields = line.split()
            if table and len(fields)>=7:
                try:
                    row = [float(x) for x in fields[:7]]
                except ValueError:
                    continue
                if all(math.isfinite(x) for x in row):
                    rows.append(row)
        if not rows:
            raise ValueError('Empty polar table '+str(path))
        peak = max(range(len(rows)),key=lambda i:rows[i][1])
        records.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(),
            reynolds=float(match[1])*10**int(match[2]),
            sampled_peak_cl=rows[peak][1], peak_alpha_deg=rows[peak][0],
            positive_peak_bracketed=0<peak<len(rows)-1,
            source_type='bundled_XFOIL_table_not_powered_wing_validation'))
    values = [v['reynolds'] for v in records]
    positive_required = math.isfinite(required_reynolds) and required_reynolds>0
    if not positive_required:
        raise ValueError('Invalid required Reynolds number')
    return dict(airfoil=airfoil, required_reynolds=required_reynolds, files=records,
                reynolds_bracketed=bool(values and min(values)<=required_reynolds<=max(values)),
                usable_as_powered_clmax=False)


def study(candidate, mass_kg, polar_folder):
    continuous, discrete = candidate['continuous'], candidate['discrete']
    area, ar, taper, tc = (continuous[k] for k in ('wing_area','aspect_ratio','taper_ratio','thickness_to_chord'))
    cl2d = 2*(tc/.12)**.15
    cl3d = cl2d/(1+cl2d/(math.pi*ar*.8))
    bounds = {v.name:(v.lower,v.upper) for v in CONTINUOUS_SPECS}
    rows = []
    # Sensitivity points, not proposed optimization candidates. Some areas fall
    # outside the current search bounds; report that instead of silently clipping.
    for fraction in (1.,.95,.9,.85):
        for count in (2,4,6,8,10,12):
            c = dict(continuous, wing_area=area*fraction, prop_span_spacing=.78,
                     prop_diameter_ratio=continuous['tip_prop_diameter']/1.72)
            d = dict(discrete, prop_layout='wingtip' if count==2 else 'leading_edge_distributed',
                     propeller_count=count)
            layout = make_layout(c,d)
            geometry_ok = all(v<=0 for v in layout['constraints'].values())
            projection = footprint(c['wing_area'],layout['span_m'],taper,
                [(y,layout['diameter_m']/2,0.) for y in layout['y_positions_m']],.6)
            requirement = lift_requirement(mass_kg,c['wing_area'],31.4,cl3d)
            power = conditional_power_requirement(requirement['required_global_loading_ratio'],
                projection['coverage_fraction'],layout['total_disk_area_m2'],31.4)
            within = bounds['wing_area'][0]<=c['wing_area']<=bounds['wing_area'][1]
            rows.append(dict(area_fraction=fraction, area_m2=c['wing_area'], count=count,
                within_current_wing_area_bounds=within, geometry_clearances_pass=geometry_ok,
                layout=layout, projection=projection, requirement=requirement,
                conditional_power=power if geometry_ok else None,
                admissible_optimization_candidate=False))
    # Root/tip values expose missing polar coverage, rather than using only MAC.
    span = math.sqrt(area*ar)
    root = 2*area/(span*(1+taper))
    reynolds = [1.225*31.4*chord/1.7894e-5 for chord in (root*taper,root)]
    polars = [polar_inventory(polar_folder,discrete['wing_airfoil'],v) for v in reynolds]
    return dict(version=VERSION, scope='fixed_mass_conditional_study_not_aircraft_optimization',
        assumptions=dict(mass_kg=mass_kg,mass_held_fixed_for_area_sweep=True,
            speed_mps=31.4,density_kg_m3=1.225,dynamic_viscosity_Pa_s=1.7894e-5,
            wing_and_disks_coplanar=True, distributed_spacing_ratio=.78,
            uniform_local_loading=True, unpowered_clmax_source='inherited_formula_at_requested_geometry_not_measured',
            requested_geometry_must_match_realized_geometry=True),
        polar_coverage=polars, scenarios=rows, ready_for_architecture_optimization=False,
        blockers=['powered_wing_CLmax_validation','separate_cruise_highlift_operating_groups',
                  'installed_component_mass_calibration','folded_rotor_drag_and_power_limits',
                  'time_and_space_discretization_convergence'])
