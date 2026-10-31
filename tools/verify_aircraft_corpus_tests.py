"""Reject partial/skipped JUnit acceptance before any research evaluations."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

REQUIRED = ('test_aircraft_corpus_protocol', 'test_aircraft_corpus_optimizer',
            'test_aircraft_corpus_pilot', 'test_structure_corpus',
            'test_structured_encoder', 'test_risk_core_mixed_gp', 'test_risk_core_observations')

def verify(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    if not cases or any(list(case.iter(tag)) for case in cases for tag in ('failure','error','skipped')):
        raise ValueError('Empty, failed, errored or skipped regression suite; stop before evaluations')
    suites = list(root.iter('testsuite'))
    if any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('Unsuccessful suite counters')
    missing = [name for name in REQUIRED if not any(name in c.get('classname','') for c in cases)]
    if missing:
        raise ValueError('Required regression modules missing: '+', '.join(missing))
    return {'testcases':len(cases), 'failures':0, 'errors':0, 'skipped':0,
            'required_modules':list(REQUIRED), 'research_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('junit',type=Path)
    print(verify(p.parse_args().junit))
