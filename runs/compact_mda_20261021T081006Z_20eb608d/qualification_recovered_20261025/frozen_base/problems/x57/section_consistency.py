"""Explicit opt-in synchronization of candidate and realized wing thickness.

VLM retains the selected airfoil camber family. This changes section thickness
metadata consistently, not the original coordinate/polar files or CLmax model.
"""

from functools import wraps
import math

VERSION = 'section_thickness_consistency_20261002_v1'


def normalize(settings):
    if settings != {'version':VERSION}:
        raise ValueError('Unsupported section-consistency configuration')
    return dict(settings)


def synchronize(candidate):
    wing = candidate.vehicle_configurations.base.wings.main_wing
    target = float(wing._tc_correct)
    if not math.isfinite(target) or not .09<=target<=.14:
        raise ValueError('Candidate section thickness outside the design domain')
    before = []
    for config in candidate.vehicle_configurations:
        w = config.wings.main_wing
        before.append(dict(config=config.tag, wing_tc=float(w.thickness_to_chord),
                           segment_tc=[float(s.thickness_to_chord) for s in w.segments]))
        w.thickness_to_chord = target
        if not len(w.segments):
            raise ValueError('Missing main-wing segments')
        for segment in w.segments:
            segment.thickness_to_chord = target
    return target,before


def install(nexus,diagnostics,settings):
    normalize(settings)
    native = nexus.procedure.simple_sizing
    @wraps(native)
    def sizing(candidate):
        target,before = synchronize(candidate)
        result = native(candidate)
        after = []
        for config in candidate.vehicle_configurations:
            w = config.wings.main_wing
            values = [float(w.thickness_to_chord)]+[float(s.thickness_to_chord) for s in w.segments]
            if not all(math.isclose(v,target,rel_tol=1e-10,abs_tol=1e-12) for v in values):
                raise ValueError('Geometry sizing overwrote candidate section thickness')
            after.append(dict(config=config.tag,wing_tc=values[0],segment_tc=values[1:]))
        diagnostics['section_consistency'] = dict(version=VERSION,requested_tc=target,before=before,after=after,
            coordinate_files_changed=False,polar_files_changed=False,
            powered_stall_validated=False,source='candidate_design_variable_not_NASA_calibration')
        return result
    nexus.procedure.simple_sizing = sizing
