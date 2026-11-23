"""Resumable 320-call trajectories for the three existing LSBO policies."""

from dataclasses import asdict
import importlib
from pathlib import Path
from threading import RLock

import torch

from .x57_cold_campaign import ObservationAdapter, update_native
from .x57_cold_risk import propose
from .x57_energy_stop import prefix_energy
from .x57_energy_surrogate import EnergyPosterior, energy_training_row
from .x57_native_slot import configure_native_search

ARMS = ("lsbo_full_budget", "ra_lsbo_energy_sign", "ra_lsbo_energy_interval")
TOTAL_BUDGET = 320
CHECKPOINTS = {
    8: (8, 16, 24, 32, 48, 64, 128, 256, 320),
    32: (32, 48, 64, 128, 256, 320),
    64: (64, 72, 96, 128, 192, 256, 320),
}
TORCH_LOCK = RLock()


class FormalNative:
    """Separate paid dispatch from the native optimizer's sequential updates."""

    def __init__(self, problem, arm, seed, initial, output):
        if arm not in ARMS or len(initial) not in CHECKPOINTS:
            raise ValueError("Unsupported formal arm or shared initialization size")
        self.problem, self.arm, self.seed, self.output = problem, arm, seed, Path(output)
        self.initial, self.online, self.history = [], [], []
        self.distance = problem.config["mission"]["cruise_distance_nmi"]
        self.native = importlib.import_module("lsbo.optimizer")
        self.settings = self.native.Settings(seed=seed, initial=len(initial), budget=TOTAL_BUDGET)
        self.pending = None
        self.native_stop_positions = []
        self.output.mkdir(parents=True, exist_ok=False)
        with TORCH_LOCK, torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (1 << 63))
            adapter = ObservationAdapter(problem.problem, self.native.Evaluation, native_feedback=arm == ARMS[0])
            self.optimizer = self.native.Optimizer(adapter, self.settings, self.output)
            if arm != ARMS[0]:
                configure_native_search(self.optimizer, reserve=True, joint_final=True, force_weighted=True)
            for row in initial:
                self.initial.append(self._accept(row, "initial"))

    def _accept(self, row, phase):
        position = len(self.history) + 1
        candidate = self.problem.candidate_class(**row["candidate"])
        if (row["feedback"]["outcome"] == "infrastructure_error" or row["charged"] is not True
                or row["candidate_id"] != self.problem.space.key(candidate) or row["evaluation"] != position
                or row["seed"] != self.seed):
            raise ValueError("Expected the next settled worker observation")
        if update_native(self.optimizer, candidate, row, phase, row["proposal"]):
            self.native_stop_positions.append(position)
        self.history.append(row)
        return row

    def _fit_energy(self, representation, bounds):
        evidence = [energy_training_row(row, f"{row['evaluation']}:{row['candidate_id']}", self.distance,
                                       prefix_energy, self.arm == ARMS[2])[0] for row in self.history]
        return EnergyPosterior(representation, [obs.candidate for obs in self.optimizer.observations],
                               evidence, self.seed, bounds)

    def ask(self):
        if self.pending is not None or len(self.history) >= TOTAL_BUDGET:
            raise RuntimeError("Settle the pending point before asking within the 320-call budget")
        self.problem.verify_frozen()
        with TORCH_LOCK, torch.random.fork_rng(devices=[]):
            torch.manual_seed((self.seed + len(self.online) + 1) % (1 << 63))
            point, phase, proposal = propose(
                self.optimizer, self.history, self.arm if self.arm == ARMS[0] else "structured_sign",
                self.native.mixed_candidate_pool, 512, fit_energy=None if self.arm == ARMS[0] else self._fit_energy)
        self.pending = (self.problem.space.key(point), phase)
        return point, phase, proposal

    def tell(self, row):
        if (self.pending is None or self.pending != (row["candidate_id"], row["phase"])
                or row["arm"] != self.arm or row["online"] != len(self.online) + 1):
            raise ValueError("Worker feedback must match the pending proposal")
        with TORCH_LOCK:
            before = len(self.native_stop_positions)
            self.online.append(self._accept(row, row["phase"]))
            self.pending = None
            return len(self.native_stop_positions) > before

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Only settled boundaries can be checkpointed")
        return {"arm": self.arm, "settings": asdict(self.settings), "contract_id": self.problem.contract_id,
                "initial": list(self.initial), "online": list(self.online),
                "native_stop_positions": list(self.native_stop_positions)}

    @classmethod
    def resume(cls, problem, checkpoint, output):
        if checkpoint["contract_id"] != problem.contract_id:
            raise ValueError("Resume requires the same physical contract")
        optimizer = cls(problem, checkpoint["arm"], checkpoint["settings"]["seed"], checkpoint["initial"], output)
        if asdict(optimizer.settings) != checkpoint["settings"]:
            raise ValueError("Resume requires the same 320-call optimizer settings")
        for row in checkpoint["online"]:
            optimizer.ask()
            optimizer.tell(row)
        if optimizer.native_stop_positions != checkpoint["native_stop_positions"]:
            raise ValueError("Replay changed the native missing objective stop positions")
        return optimizer
