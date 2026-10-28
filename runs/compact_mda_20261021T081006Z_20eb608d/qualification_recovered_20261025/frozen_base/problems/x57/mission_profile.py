"""Versioned conceptual mission assumptions, not a NASA flight-test schedule."""

import math

PROFILE_VERSION = 'concept_mission_20260930_v3'
LEGACY_PROFILE_VERSION = 'concept_mission_20260926_v2'
SHORT_PROFILE_VERSION = 'concept_mission_20261004_v4'
DEFAULT_SETTINGS = {'version': PROFILE_VERSION, 'main_descent_rate_ft_min': 450.,
                    'downleg_distance_ft': 6000., 'baseleg_descent_rate_ft_min': 250.}
FT = .3048


def normalize_settings(settings=None):
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    short_profile = settings.get('version') == SHORT_PROFILE_VERSION
    expected = set(DEFAULT_SETTINGS) | ({'cruise_distance_nmi'} if short_profile else set())
    if (set(settings) != expected
            or settings['version'] not in (PROFILE_VERSION, LEGACY_PROFILE_VERSION, SHORT_PROFILE_VERSION)):
        raise ValueError('Unsupported mission profile; use the current versioned mission config')
    for key in sorted(expected - {'version'}):
        if isinstance(settings[key], bool):
            raise ValueError(f'Invalid mission setting: {key}')
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'Invalid mission setting: {key}')
    return settings


def configure_mission(nexus, settings=None):
    settings = normalize_settings(settings)
    mission = nexus.missions.base_mission
    if getattr(mission, '_lsbo_mission_profile_installed', False):
        raise ValueError('Mission profile already installed')
    segments = {str(segment.tag).lower(): segment for segment in mission.segments.values()}
    descent, cruise, downleg = (segments[name] for name in ('decent', 'cruise', 'downleg'))
    rate = settings['main_descent_rate_ft_min'] * FT / 60.
    v0, vf = float(cruise.air_speed), float(descent.air_speed_end)
    height = float(descent.altitude_start - descent.altitude_end)
    if height <= 0 or min(v0, vf) <= rate:
        raise ValueError('Invalid descent height, speed, or rate')
    distance = settings['downleg_distance_ft'] * FT
    level_end = float(downleg.air_speed_end)
    if not 0 < level_end < vf:
        raise ValueError('Downleg must decelerate from descent-end speed')
    acceleration = (level_end ** 2 - vf ** 2) / (2 * distance)
    old_rate = float(descent.climb_rate)
    old_acceleration = float(downleg.acceleration)
    baseleg, reserve_descent = segments['baseleg'], segments['reserve_descent']
    base_rate = settings['baseleg_descent_rate_ft_min'] * FT / 60.
    base_height = float(baseleg.altitude_start - baseleg.altitude_end)
    if base_height <= 0 or min(reserve_descent.air_speed, baseleg.air_speed_end) <= base_rate:
        raise ValueError('Invalid baseleg height, speed, or rate')
    old_base_rate = float(baseleg.climb_rate)
    old_cruise_distance = float(cruise.distance)
    if settings['version'] == SHORT_PROFILE_VERSION:
        cruise.distance = settings['cruise_distance_nmi'] * 1852.
    descent.air_speed_start = v0
    descent.climb_rate = -rate
    downleg.air_speed_start = vf
    downleg.altitude = float(descent.altitude_end)
    downleg.distance = distance
    # RCAIDE's constant-acceleration segment ignores its distance attribute;
    # enforce that intended distance through consistent acceleration and speeds.
    downleg.acceleration = acceleration
    baseleg.air_speed_start = float(reserve_descent.air_speed)
    baseleg.climb_rate = -base_rate
    mission._lsbo_mission_profile_installed = True
    return {'mission_profile': settings,
            'mission_assumption_source': 'conceptual_design_scenario_not_NASA_flight_test_schedule',
            'main_descent_duration_s': height / rate,
            'previous_main_descent_duration_s': -height / old_rate,
            'downleg_duration_s': (level_end - vf) / acceleration,
            'previous_downleg_duration_s': (level_end - vf) / old_acceleration,
            'downleg_effective_distance_m': distance,
            'downleg_acceleration_m_s2': acceleration,
            'baseleg_duration_s': base_height / base_rate,
            'previous_baseleg_duration_s': -base_height / old_base_rate,
            'baseleg_initial_speed_m_s': float(baseleg.air_speed_start),
            'cruise_distance_m': float(cruise.distance),
            'previous_cruise_distance_m': old_cruise_distance,
            'mission_end_altitude_m': float(segments['baseleg'].altitude_end),
            'reserve_segments_retained': True}
