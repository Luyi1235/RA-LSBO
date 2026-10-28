"""Data-bounded first-sectional-limit screen, not a validated powered CLmax."""

import hashlib
import math
from pathlib import Path
import re
import numpy as np


def read_polar(path):
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode('utf-8', errors='replace')
    re_match = re.search(r'Re\s*=\s*([\d.]+)\s*e\s*([+-]?\d+)', text)
    mach_match = re.search(r'Mach\s*=\s*([\d.]+)', text)
    if not re_match or not mach_match:
        raise ValueError('Missing Reynolds/Mach metadata: '+path.name)
    rows, table = [], False
    for line in text.splitlines():
        if '------' in line:
            table = True
            continue
        if table and len(line.split()) >= 7:
            try: row = [float(v) for v in line.split()[:7]]
            except ValueError: continue
            if not all(math.isfinite(v) for v in row):
                raise ValueError('Nonfinite polar row')
            rows.append(row)
    rows = np.asarray(rows)
    if rows.ndim != 2 or len(rows) < 4 or np.any(np.diff(rows[:,0]) <= 0):
        raise ValueError('Invalid polar alpha sequence')
    peak = int(np.argmax(rows[:,1]))
    bracketed = 0 < peak < len(rows)-1 and rows[-1,1] < rows[peak,1]
    return dict(file=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        reynolds=float(re_match[1])*10**int(re_match[2]), mach=float(mach_match[1]),
        cl_peak=float(rows[peak,1]), alpha_peak_deg=float(rows[peak,0]),
        peak_bracketed=bool(bracketed), alpha_range_deg=[float(rows[0,0]),float(rows[-1,0])])


def section_screen(section_cl, q_ratio, reynolds, polars, geometry_matches):
    cl, ratio, re = (np.asarray(x, dtype=float) for x in (section_cl,q_ratio,reynolds))
    if (cl.ndim != 1 or cl.shape != ratio.shape or cl.shape != re.shape or not len(cl)
            or not all(np.isfinite(x).all() for x in (cl,ratio,re))
            or np.any(ratio<=0) or np.any(re<=0)):
        raise ValueError('Invalid sectional load/reference arrays')
    records = sorted(polars, key=lambda p:p['reynolds'])
    good = bool(records) and all(p['peak_bracketed'] and p['mach']==0 for p in records)
    bracket = np.array([bool(records) and records[0]['reynolds']<=r<=records[-1]['reynolds'] for r in re])
    reasons = []
    if np.any(cl < -1e-8): reasons.append('negative_lift_limit_not_available')
    if not geometry_matches: reasons.append('wing_thickness_does_not_match_polar_geometry')
    if not good: reasons.append('missing_or_unbracketed_polar_peak')
    if not np.all(bracket): reasons.append('Reynolds_outside_polar_range_no_extrapolation')
    result = dict(status='unknown' if reasons else 'screen_only_not_validated_stall', reasons=reasons,
        reynolds_min=float(re.min()), reynolds_max=float(re.max()),
        uncovered_sections=int((~bracket).sum()), section_count=len(cl),
        required_local_cl=(cl/ratio).tolist(), minimum_section_margin=None,
        section_limit_passed=None)
    if not reasons:
        limit = np.interp(np.log(re), np.log([p['reynolds'] for p in records]),
                          [p['cl_peak'] for p in records])
        margin = limit-cl/ratio
        result.update(minimum_section_margin=float(margin.min()),
                      section_limit_passed=bool(np.all(margin>=0)),
                      section_cl_limit=limit.tolist())
    return result


def screen_envelope(rows, required_cl):
    """Never turn unknown sectional data into a feasible design."""
    if not math.isfinite(required_cl) or required_cl <= 0:
        raise ValueError('Invalid required CL')
    eligible = [r for r in rows if r['section_screen']['section_limit_passed'] is True
                and r['electrical']['within_reference_envelope'] is True]
    if not eligible:
        return dict(status='undetermined_no_data_supported_power_admissible_sample',
                    sampled_lift_margin=None, eligible_samples=0,
                    constraint_override_permitted=False)
    best = max(eligible, key=lambda r:r['CL'])
    return dict(status='sampled_untrimmed_section_screen_NOT_CLmax', eligible_samples=len(eligible),
                sampled_lift_margin=best['CL']-required_cl, selected_alpha_deg=best['alpha_deg'],
                selected_rpm_ratio=best['rpm_ratio'], constraint_override_permitted=False)
