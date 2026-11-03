"""Shared proposal interface and explicitly simulated adapters, never real algorithms."""
from dataclasses import dataclass
from .ledger import digest

QUALIFICATION = {
    'ra_lsbo': ('BLOCKED', 'output/evidence-sensitive method not frozen or qualified'),
    'full_latent': ('BLOCKED', 'historical corpus optimizer exists; confirmation adapter not qualified'),
    'full_onehot': ('BLOCKED', 'historical one-hot optimizer exists; confirmation adapter not qualified'),
    'reverse_representation': ('BLOCKED', 'candidate view routing exists; full adapter and fallback not qualified'),
    'interval_as_missing': ('BLOCKED', 'ablation semantics not frozen or qualified'),
    'random': ('BLOCKED', 'historical random exists; common q_A and call contract not qualified'),
    'cbob_adapted': ('BLOCKED', 'CBOB reference, mixed-variable and feedback adaptation not qualified'),
}


@dataclass(frozen=True)
class Trial:
    case: str
    seed: int
    arm: str
    max_online: int = 18


class SimulationAdapter:
    """Identical synthetic test policy under seven labels, not algorithm implementations."""
    def __init__(self, trial):
        if trial.arm not in QUALIFICATION or trial.max_online != 18:
            raise ValueError('unregistered simulated arm or budget')
        self.trial = trial

    def propose(self, history, pool):
        if not pool:
            raise ValueError('empty simulated pool')
        online = 0
        for record in history:
            if (record.get('case'), record.get('seed')) != (self.trial.case, self.trial.seed):
                raise ValueError('foreign case or seed in history')
            if record.get('role') == 'initial':
                if record.get('arm') != '':
                    raise ValueError('shared initial must not belong to an arm')
            elif record.get('role') == 'online':
                if record.get('arm') != self.trial.arm:
                    raise ValueError('other arm observation in history')
                online += 1
            else:
                raise ValueError('unknown or verification data in optimization history')
        if online >= self.trial.max_online:
            raise RuntimeError('simulated online budget exhausted')
        index = int(digest({'trial': self.trial.__dict__, 'history': history})[:8], 16) % len(pool)
        return pool[index], {'simulated_only': True, 'online_index': online}


def create(arm, trial, *, scope):
    if arm not in QUALIFICATION or trial.arm != arm:
        raise ValueError('unregistered or mismatched arm')
    if scope != 'simulation':
        raise RuntimeError('real algorithm adapter is not qualified: ' + QUALIFICATION[arm][1])
    return SimulationAdapter(trial)
