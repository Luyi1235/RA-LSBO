"""Native BEM/VLM paired diagnostics with data- and power-bounded lift screening."""

from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
import math
from pathlib import Path
import traceback
import numpy as np
from .sectional_limits import read_polar, section_screen, screen_envelope

VERSION = 'native_powered_lift_probe_20261002_v1'
DEFAULTS = dict(version=VERSION, speed_mps=31.4, soc=.2,
                alpha_deg=[0.,4.,8.,12.,16.], rpm_ratios=[.6,.8,1.],
                spanwise_panels=[12,24,36], chordwise_panels=5)


def normalize(settings):
    if not isinstance(settings,dict) or set(settings)!=set(DEFAULTS) or settings['version']!=VERSION:
        raise ValueError('Explicit versioned powered-lift probe settings required')
    s=deepcopy(settings)
    for k,lo,hi in (('speed_mps',15,60),('soc',.15,1)):
        if type(s[k]) not in (int,float) or not math.isfinite(s[k]) or not lo<=s[k]<=hi:
            raise ValueError('Invalid probe '+k)
    for k,lo,hi in (('alpha_deg',-5,20),('rpm_ratios',.1,1)):
        x=s[k]
        if (not isinstance(x,list) or not x or len(x)>20 or any(type(v) not in (int,float)
                or not math.isfinite(v) or not lo<=v<=hi for v in x) or x!=sorted(set(x))):
            raise ValueError('Invalid probe '+k)
    if (not isinstance(s['spanwise_panels'],list) or len(s['spanwise_panels'])<2
            or s['spanwise_panels']!=sorted(set(s['spanwise_panels']))
            or any(type(v) is not int or not 8<=v<=64 for v in s['spanwise_panels'])
            or type(s['chordwise_panels']) is not int or not 2<=s['chordwise_panels']<=10):
        raise ValueError('Invalid probe panel refinement')
    return s


def first_row(data, count):
    result=deepcopy(data)
    def reduce(value):
        for key,item in list(value.items()):
            if isinstance(item,np.ndarray) and item.ndim and item.shape[0]==count:
                value[key]=item[:1].copy()
            elif isinstance(item,dict): reduce(item)
    reduce(result)
    return result


def flight_conditions(seed, velocity, alpha_deg):
    from RCAIDE.Framework.Core import angles_to_dcms, orientation_transpose
    c=first_row(seed,len(seed.freestream.velocity))
    # Explicit sea-level diagnostic atmosphere, not a mission-state replacement.
    for key,value in dict(velocity=velocity,density=1.225,dynamic_viscosity=1.7894e-5,
            temperature=288.15,speed_of_sound=340.294,mach_number=velocity/340.294,
            dynamic_pressure=.5*1.225*velocity**2).items():
        setattr(c.freestream,key,np.array([[value]]))
    a=math.radians(alpha_deg)
    c.aerodynamics.angles.alpha=np.array([[a]])
    c.aerodynamics.angles.beta=np.zeros((1,1))
    c.frames.inertial.velocity_vector=np.array([[velocity,0.,0.]])
    c.frames.body.inertial_rotations=np.array([[0.,a,0.]])
    c.frames.body.transform_to_inertial=orientation_transpose(angles_to_dcms(c.frames.body.inertial_rotations,(2,1,0)))
    for key in ('pitch_rate','roll_rate','yaw_rate'): setattr(c.static_stability,key,np.zeros((1,1)))
    return c


def drive_power(props,c,soc,auxiliary_W):
    from .nasa_battery import electrical_state,TOTAL_CELLS,BatteryDomainError,CELL_MAX_A,CELL_CUTOFF_V
    records=[]
    for p in props:
        r=c.energy.converters[p.rotor.tag]
        omega,torque,power=(float(np.asarray(r[k]).reshape(-1)[0]) for k in ('omega','torque','power'))
        m=p.motor
        if (not math.isclose(power,torque*omega,rel_tol=1e-7,abs_tol=1e-5)
                or not 0<float(p.electronic_speed_controller.efficiency)<=1
                or float(m.speed_constant)<=0 or float(m.resistance)<0
                or float(m.no_load_current)<0):
            raise ValueError('Inconsistent shaft power or invalid DC motor/ESC parameters')
        if not math.isclose(float(m.gearbox.gear_ratio),1.):
            raise ValueError('Probe DC inverse currently requires the installed direct-drive motors')
        current=torque*float(m.speed_constant)+float(m.no_load_current)
        voltage=omega/float(m.speed_constant)+current*float(m.resistance)
        electrical=voltage*current/float(p.electronic_speed_controller.efficiency)
        if min(torque,power,current,voltage,electrical)<0 or not all(math.isfinite(v) for v in (torque,power,current,voltage,electrical)):
            raise ValueError('Probe reached windmilling/nonfinite motor state; not a powered-lift point')
        records.append(dict(tag=p.tag,shaft_W=power,torque_Nm=torque,motor_current_A=current,
                            motor_voltage_V=voltage,esc_input_W=electrical))
    total=sum(r['esc_input_W'] for r in records)+auxiliary_W
    result=dict(terminal_power_W=total,shaft_power_W=sum(r['shaft_W'] for r in records),
                components=records,soc=soc,within_reference_envelope=False,
                motor_thermal_rating_validated=False)
    try:
        current,voltage,_,_=electrical_state(soc,total/TOTAL_CELLS,enforce_limits=False)
        throttle=max(r['motor_voltage_V']/(voltage*128) for r in records)
        result.update(cell_current_A=current,pack_voltage_V=voltage*128,
            required_max_throttle=throttle,
            within_reference_envelope=bool(current<=CELL_MAX_A and voltage>=CELL_CUTOFF_V and throttle<=1))
    except BatteryDomainError as error: result['domain_error']=str(error)
    return result


def summarize_vlm(result,c,wing,polars,geometry_matches):
    vd=result.VD
    # This aircraft stores the symmetric main wing first; do not silently map a tail to wing polars.
    n=int(vd.n_sw[0]+vd.n_sw[1])
    breaks=np.asarray(vd.chordwise_breaks,dtype=int)[:n]
    widths=np.asarray(vd.panels_per_strip,dtype=int)[breaks]
    vx=np.array([result.V_x[0,i:i+w].mean() for i,w in zip(breaks,widths)])
    vz=np.array([result.V_z[0,i:i+w].mean() for i,w in zip(breaks,widths)])
    v=float(c.freestream.velocity[0,0]); a=float(c.aerodynamics.angles.alpha[0,0])
    local_speed=np.sqrt((v*np.cos(a)+vx)**2+(v*np.sin(a)+vz)**2)
    ratio=(local_speed/v)**2
    chord=np.asarray(vd.chord_lengths).reshape(-1)[breaks]
    reynolds=float(c.freestream.density[0,0])*local_speed*chord/float(c.freestream.dynamic_viscosity[0,0])
    section=np.asarray(result.sectional_CLift)[0,:n]
    return dict(CL=float(result.CLift[0,0]),CDi=float(result.CDrag_induced[0,0]),
        main_wing_CL=float(result.CLift_wings[wing.tag][0,0]),
        span_y_m=np.asarray(result.spanwise_stations).reshape(-1)[:n].tolist(),
        sectional_cl_qinf=section.tolist(),local_q_ratio=ratio.tolist(),
        reynolds=reynolds.tolist(),
        max_axial_induced_mps=float(np.max(result.V_x)),
        section_screen=section_screen(section,ratio,reynolds,polars,geometry_matches))


def run_probe(candidate,evaluator,settings,report,save):
    from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.VLM import VLM
    from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.compute_rotor_performance import compute_rotor_performance
    from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake import wake_model
    from .propulsion_research import rowwise_axial_wake
    vehicle=deepcopy(candidate.vehicle_configurations.base)
    if list(vehicle.wings.values())[0].tag!='main_wing' or not vehicle.wings.main_wing.symmetric:
        raise ValueError('Unsupported wing ordering or asymmetric wing')
    wing=vehicle.wings.main_wing
    props=list(evaluator._network(vehicle).propulsors)
    if any(not p.active for p in props): raise ValueError('This probe requires all rotors active')
    bus=next(iter(evaluator._network(vehicle).busses.values()))
    if '_powered_lift_seed' in candidate:
        seed=candidate._powered_lift_seed
        auxiliary=float(bus.avionics.power_draw)+float(bus.payload.power_draw)
        if not math.isclose(float(bus.efficiency),1.):
            raise ValueError('Pointwise diagnostic requires the reference unit-efficiency bus')
    else:
        seed=next(iter(candidate.results.base.segments.values())).conditions
        esc=sum(float(seed.energy.modulators[p.electronic_speed_controller.tag].inputs.power[0,0]) for p in props)
        auxiliary=max(0.,float(seed.energy[bus.tag].power_draw[0,0])-esc)
    polar_dir=Path(__file__).parent/'model/Airfoils/Polars'
    # Match actual assigned airfoil rather than inferring it from thickness alone.
    families={str(section.airfoil.coordinate_file):section.airfoil for section in wing.segments
              if getattr(section,'airfoil',None) is not None}
    airfoils=list(families.values())
    if len(airfoils)!=1: raise ValueError('Probe needs one known main-wing airfoil family')
    filename=Path(airfoils[0].coordinate_file).stem
    polars=[read_polar(p) for p in sorted(polar_dir.glob(filename+'_polar_Re_*.txt'))]
    coordinate=Path(airfoils[0].coordinate_file)
    from RCAIDE.Library.Methods.Geometry.Airfoil import import_airfoil_geometry
    polar_tc=float(import_airfoil_geometry(str(coordinate)).thickness_to_chord)
    matches=abs(float(wing.thickness_to_chord)-polar_tc)<=.003
    required=2*float(vehicle.mass_properties.max_takeoff)*9.81/(1.225*settings['speed_mps']**2*float(vehicle.reference_area))
    report.update(polars=polars,airfoil=filename,coordinate_thickness=polar_tc,
        wing_thickness=float(wing.thickness_to_chord),geometry_matches_polars=matches,
        required_CL=required,mass_kg=float(vehicle.mass_properties.max_takeoff),
        reference_area_m2=float(vehicle.reference_area),auxiliary_W=auxiliary,
        native_source_sha256={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in (VLM,compute_rotor_performance,wake_model.compute_wake_induced_velocity)},rows=[])
    native_wake=wake_model.compute_wake_induced_velocity
    mode='disabled'
    def axial(rotor,conditions,points,count):
        if mode=='zero': return np.zeros((count,len(points.XC),3))
        return rowwise_axial_wake(native_wake,rotor,conditions,points,count)
    wake_model.compute_wake_induced_velocity=axial
    try:
        for alpha in settings['alpha_deg']:
            for rpm_ratio in settings['rpm_ratios']:
                c=flight_conditions(seed,settings['speed_mps'],alpha)
                for p in props:
                    rc=c.energy.converters[p.rotor.tag]
                    rc.omega=np.array([[float(p.rotor.cruise.design_angular_velocity)*rpm_ratio]])
                    rc.throttle=np.ones((1,1))
                    rc.design_flag=False
                    rc.commanded_thrust_vector_angle=np.zeros((1,1))
                    compute_rotor_performance(p.rotor,c)
                electrical=drive_power(props,c,settings['soc'],auxiliary)
                for panels in settings['spanwise_panels']:
                    options=deepcopy(candidate.analyses.base.aerodynamics.settings)
                    options.number_of_spanwise_vortices=panels
                    options.number_of_chordwise_vortices=settings['chordwise_panels']
                    paired={}
                    for mode in ('disabled','zero','axial'):
                        options.propeller_wake_model=mode!='disabled'
                        result=VLM(c,options,vehicle)
                        paired[mode]=summarize_vlm(result,c,wing,polars,matches)
                    difference=abs(paired['zero']['CL']-paired['disabled']['CL'])
                    if difference>1e-8: raise ValueError('Zero-wake identity check failed')
                    row=dict(alpha_deg=alpha,rpm_ratio=rpm_ratio,spanwise_panels=panels,
                        chordwise_panels=settings['chordwise_panels'],electrical=electrical,
                        zero_wake_delta_CL=difference,
                        delta_CL=paired['axial']['CL']-paired['disabled']['CL'],
                        delta_CDi=paired['axial']['CDi']-paired['disabled']['CDi'],paired=paired)
                    report['rows'].append(row)
                    save()
                    print(f'[POWERED LIFT] alpha={alpha:g}, rpm_ratio={rpm_ratio:g}, panels={panels}, '
                          f'dCL={row["delta_CL"]:.6g}, power_ok={electrical["within_reference_envelope"]}',flush=True)
        last=settings['spanwise_panels'][-1]
        eligible=[dict(r['paired']['axial'],alpha_deg=r['alpha_deg'],rpm_ratio=r['rpm_ratio'],electrical=r['electrical'])
                  for r in report['rows'] if r['spanwise_panels']==last]
        report['low_speed_screen']=screen_envelope(eligible,required)
        report['status']='completed'
        save()
    finally:
        wake_model.compute_wake_induced_velocity=native_wake


def install(nexus,evaluator,diagnostics,settings,artifact_dir):
    settings=normalize(settings)
    physics=diagnostics.get('propulsion_research',{}).get('settings',{})
    if physics.get('native_wake',True) or 'dep_group_schedule' in diagnostics:
        raise ValueError('Probe requires wake-off reference mission with all propulsors active')
    report=dict(version=VERSION,settings=settings,status='not_run',
        scope='fixed_RPM_untrimmed_native_axial_wake_sensitivity_NOT_powered_CLmax',
        powered_stall_credit_applied=False,ready_for_optimization=False,
        limitations=['no_swirl','native_lateral_wake_interpolation_not_3D_contraction',
                     'no_wing_to_propeller_upwash','no_viscous_separation_prediction',
                     'sectional_limit_screen_requires_validation_and_trim',
                     'configured_design_RPM_not_verified_motor_thermal_rating'])
    from .airfoil_assignment import install as assign_airfoil
    assign_airfoil(evaluator,report)
    def save():
        if artifact_dir is not None:
            path=Path(artifact_dir)/'powered_lift_probe.json'
            path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            diagnostics['powered_lift_probe_file']=str(path)
        diagnostics['powered_lift_probe']=dict(status=report['status'],version=VERSION,
            row_count=len(report.get('rows',[])),low_speed_screen=report.get('low_speed_screen'),
            failure=report.get('failure'),powered_stall_credit_applied=False,ready_for_optimization=False)
    def execute(candidate):
        try: run_probe(candidate,evaluator,settings,report,save)
        except Exception as error:
            traceback.print_exc()
            report.update(status='failed',failure=f'{type(error).__name__}: {error}')
            save()
    native=nexus.procedure.post_process
    @wraps(native)
    def post(candidate):
        execute(candidate)
        # Preserve every original mission audit; a pointwise aerodynamic probe
        # may succeed even when the complete mission violates battery limits.
        return native(candidate)
    nexus.procedure.post_process=post
    return execute


def geometry_only(nexus,evaluator,candidate,execute,diagnostics):
    """Reuse candidate construction/sizing without pretending to run a mission."""
    from lsbo.records import Evaluation
    original=nexus.procedure
    captured=[]
    def process(c):
        original.simple_sizing(c)
        original.weights(c)
        segment=next(iter(c.missions.base_mission.segments.values()))
        for prop in evaluator._network(c.vehicle_configurations.base).propulsors:
            prop.append_operating_conditions(segment,segment.state.conditions.energy,segment.state.conditions.noise)
        c._powered_lift_seed=segment.state.conditions
        execute(c)
        captured.append(True)
    nexus.procedure=process
    raw=evaluator.evaluate_candidate(nexus,candidate['continuous'],candidate['discrete'])
    # The preserved evaluator correctly refuses a mission objective without a
    # converged mission. Only the separate diagnostic result is exported here.
    if not captured or raw.status!='mission_not_converged':
        raise RuntimeError('Diagnostic geometry construction failed: '+raw.status)
    diagnostics['mda_evaluated']=False
    diagnostics['mission_evaluated']=False
    diagnostics['constraint_override_permitted']=False
    return Evaluation(None,False,status='diagnostic_only_not_optimization_data',metrics=diagnostics)
