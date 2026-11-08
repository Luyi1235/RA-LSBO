"""A read-only audit must not activate the paid B1 evaluator workflow."""
from pathlib import Path
import fnmatch
import re
import unittest


class TriggerGuard(unittest.TestCase):
    def test_readonly_paths_do_not_match_old_evaluator_trigger(self):
        root=Path(__file__).resolve().parents[1]
        old=(root/'.github/workflows/b1-objective-value.yml').read_text(encoding='utf8')
        patterns=re.findall(r"^      - '([^']+)'$",old.split('permissions:',1)[0],re.M)
        self.assertTrue(patterns)
        paths=('pr78_bottleneck_core.py','run_b1_bottleneck_audit.py',
               'tests/test_pr78_bottleneck.py','tests/test_pr78_trigger_guard.py',
               '.github/workflows/b1-bottleneck-audit.yml',
               'docs/PHASE_B1_BOTTLENECK_AUDIT.txt','PR78_RESULTS_AND_NEXT.txt')
        self.assertFalse([(p,g) for p in paths for g in patterns if fnmatch.fnmatchcase(p,g)])


if __name__=='__main__':
    unittest.main()
