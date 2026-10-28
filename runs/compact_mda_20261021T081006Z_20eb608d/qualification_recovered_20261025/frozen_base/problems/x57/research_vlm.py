"""Force-only direct VLM for the opt-in, force-trimmed axial-wake study.

Uses the installed VLM solver once per batch, omitting unused stability
derivative perturbations. No powered-stall or swirl benefit is assumed.
"""


def force_coefficients(state, settings, vehicle):
    from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.VLM import VLM
    result = VLM(state.conditions, settings, vehicle)
    write_coefficients(state.conditions, result)


def write_coefficients(conditions, result):
    aero = conditions.aerodynamics
    aero.coefficients.lift.induced.inviscid_wings = result.CLift_wings
    aero.coefficients.lift.compressible_wings = result.CLift_wings
    aero.coefficients.drag.induced.inviscid_wings = result.CDrag_induced_wings
    aero.coefficients.lift.induced.spanwise = result.sectional_CLift
    aero.coefficients.drag.induced.spanwise = result.sectional_CDrag_induced
    aero.coefficients.surface_pressure = result.CP
    aero.coefficients.lift.total = result.CLift
    aero.coefficients.drag.induced.inviscid = result.CDrag_induced
    aero.angles.induced = result.alpha_induced
    aero.chord_sections = result.chord_sections
    aero.spanwise_stations = result.spanwise_stations
    for dest, source in (('X','CX'),('Y','CY'),('Z','CZ'),('L','CL'),('M','CM'),('N','CN')):
        setattr(conditions.static_stability.coefficients, dest, getattr(result, source))


def install(nexus, diagnostics):
    from RCAIDE.Framework.Analyses.Aerodynamics.Vortex_Lattice_Method import Vortex_Lattice_Method
    for segment in nexus.missions.base_mission.segments.values():
        dynamics = segment.flight_dynamics
        if any(bool(getattr(dynamics, field, False)) for field in
               ('force_y','moment_x','moment_y','moment_z')):
            raise ValueError('Research force-only VLM is not a stability/moment-trim analysis')
    for analysis in nexus.analyses:
        for wing in analysis.aerodynamics.vehicle.wings:
            if len(wing.control_surfaces):
                raise ValueError('Research force-only VLM does not support moving control surfaces')
    native = Vortex_Lattice_Method.initialize
    def initialize(self, *args, **kwargs):
        result = native(self, *args, **kwargs)
        if not self.settings.use_surrogate:
            self.process.compute.lift.inviscid_wings = force_coefficients
        return result
    Vortex_Lattice_Method.initialize = initialize
    diagnostics['propulsion_research']['force_kernel'] = 'native_VLM_batch_no_stability_derivative_perturbations'
