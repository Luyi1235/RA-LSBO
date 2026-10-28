"""Worker-local compatibility fix for RCAIDE's singular segment.airfoil field."""

from copy import deepcopy
from functools import wraps
import hashlib
from pathlib import Path


def install(evaluator, report):
    original = evaluator._apply_wing_airfoil

    @wraps(original)
    def assign(config, name):
        import RCAIDE
        original(config, name)
        path = Path(__file__).parent/'model/Airfoils'/({'Clark_Y': 'Clark_y'}.get(name, name)+'.txt')
        if not path.is_file():
            raise ValueError('Missing selected wing airfoil')
        airfoil = RCAIDE.Library.Components.Airfoils.Airfoil()
        airfoil.tag = name
        airfoil.coordinate_file = str(path.resolve())
        wing = config.wings.main_wing
        wing.append_airfoil(deepcopy(airfoil))
        for section in wing.segments:
            section.append_airfoil(deepcopy(airfoil))
            if Path(section.airfoil.coordinate_file).resolve() != path.resolve():
                raise ValueError('VLM section airfoil differs from selected family')
        report['airfoil_assignment'] = dict(requested=name, reader_field='segment.airfoil',
            coordinate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            segment_count=len(wing.segments), verified=True)
    evaluator._apply_wing_airfoil = assign
