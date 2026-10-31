"""Diagnostic representation refits; no evaluator calls and no BO replay.

30 epochs is the registered budget; 120 is a fixed sensitivity diagnostic, not
an adopted optimizer setting. Grammar fit is not a generalization score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.problems import AircraftCase, MathematicalCase
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft, HierarchicalBranin
from ra_lsbo.partial_bo.records import DesignRecord

ARCHIVE_SHA = '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
PREFIX = 'aircraft_nominal_20261030_01/pilot/'


def blocked(*args, **kwargs):
    raise AssertionError('Evaluation forbidden in representation diagnosis')


def grammar_metrics(features, problem, prop):
    configs = problem.space.configurations()
    candidates = []
    for config in configs:
        raw = problem.space.default().to_dict()
        raw['discrete'] = config
        candidates.append(problem.canonical(raw))
    x = features.transform(candidates)
    nc = len(problem.space.continuous)
    block = x[:, 2*nc:]
    distances = np.sqrt(np.mean((block[:, None, :]-block[None, :, :])**2, axis=-1))
    upper = np.triu_indices(len(configs), 1)
    pairs = [(i, j) for i, c in enumerate(configs) for j, d in enumerate(configs) if i < j
             and c['prop_layout'] == d['prop_layout'] and c['propeller_count'] == d['propeller_count']]
    result = {'configurations': configs, 'feature_dimension': x.shape[1],
              'discrete_rms_distances': distances[upper].tolist(),
              'battery_pair_rms_distances': [float(distances[i, j]) for i, j in pairs],
              'feature_identity': features.identity, 'encoder_state': features.state_hash,
              'saved_feature_identity': prop['metadata'].get('feature_identity'),
              'original_identity_matches': features.identity == prop['metadata'].get('feature_identity')}
    if features.latent:
        with torch.no_grad():
            indices = features.encoder._indices([features.candidate(c) for c in candidates])
            mu, _ = features.encoder.model.encode(indices)
            mask = features.encoder.model.discrete_mask(indices).bool()
            decoded = torch.stack([layer(mu).argmax(-1) for layer in features.encoder.model.decoder], -1)
            correct = (decoded == indices) | ~mask
            result['active_grammar_exact_fraction'] = float(correct.all(-1).double().mean())
            result['active_variable_accuracy'] = {v.name: float(correct[:, i][mask[:, i]].double().mean())
                                                  for i, v in enumerate(problem.space.discrete)}
            result['latent_span'] = np.ptp(mu.numpy(), axis=0).tolist()
            result['normalization_offset'] = features.offset.tolist()
            result['normalization_scale'] = features.scale.tolist()
        result['last_training_epoch'] = features.encoder.training_trace[-1]
    corr = {}
    for name, head in prop['metadata'].get('heads', {}).items():
        if head is None or 'selected' not in head:
            continue
        length = head['selected']['discrete_length']
        d = np.asarray(result['battery_pair_rms_distances'])/length
        k = (1+np.sqrt(5)*d+5*d*d/3)*np.exp(-np.sqrt(5)*d)
        corr[name] = {'saved_discrete_length': length, 'same_layout_battery_pair_kernel': k.tolist()}
    result['kernel_diagnostic'] = corr
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use new output file')
    sha = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            sha.update(chunk)
    if sha.hexdigest() != ARCHIVE_SHA:
        raise ValueError('Input archive identity mismatch')
    for cls in (AircraftCase, MathematicalCase, SimplifiedElectricAircraft, HierarchicalBranin):
        cls.evaluate = blocked
    torch.set_num_threads(1)
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        load = lambda relative: json.loads(archive.read(PREFIX+relative))
        plan = load('plan.json')
        # Compare across CRLF/LF without altering sealed bytes or executing snapshots.
        repo = Path(__file__).resolve().parents[1]
        checked = 0
        for old, identity in plan['sources'].items():
            path = old.replace('\\', '/')
            relative = None
            for marker in ('/ra_lsbo/', '/runs/compact_mda_'):
                if marker in path:
                    relative = path[path.index(marker)+1:]
                    break
            if relative is not None and relative.endswith('.py'):
                original = archive.read(PREFIX+'source_snapshot/'+identity+'.source')
                if hashlib.sha256(original).hexdigest() != identity:
                    raise ValueError('Changed sealed source')
                if original.replace(b'\r\n', b'\n') != (repo/relative).read_bytes().replace(b'\r\n', b'\n'):
                    raise ValueError('Diagnostic implementation differs: '+relative)
                checked += 1
        for case in plan['cases']:
            problem = AircraftCase(BASE, case.split('_')[1])
            for seed in plan['profile']['seeds']:
                trial = f'optimization/{case}/{seed}_ev_latent/'
                manifest = load(trial+'manifest.json')
                online = [json.loads(line) for line in archive.read(PREFIX+trial+'observations.jsonl').splitlines()]
                props = [json.loads(line) for line in archive.read(PREFIX+trial+'proposals.jsonl').splitlines()]
                for iteration in (0, 29):
                    records = [DesignRecord.from_dict(r) for r in manifest['initial']+online[:iteration]]
                    if digest([r.to_dict() for r in records]) != props[iteration]['prefix_digest']:
                        raise ValueError('History prefix mismatch')
                    # A non-evaluator metadata proxy retains the historical contract
                    # after source-text checks. The original records remain unchanged.
                    fit_problem = SimpleNamespace(space=problem.space,
                                                  contract_id=records[0].observation.contract_id)
                    for epochs in (30, 120):
                        features = PrefixFeatures(fit_problem, BASE, records, seed, latent=True, epochs=epochs)
                        metrics = grammar_metrics(features, problem, props[iteration])
                        metrics.update(case=case, seed=seed, iteration=iteration, epochs=epochs,
                                       training_count=len(records), representation='latent',
                                       comparison='same_history_training_budget_sensitivity_not_BO_replay')
                        rows.append(metrics)
                    features = PrefixFeatures(fit_problem, BASE, records, seed, latent=False, epochs=30)
                    metrics = grammar_metrics(features, problem, props[iteration])
                    metrics.update(case=case, seed=seed, iteration=iteration, epochs=0,
                                   training_count=len(records), representation='onehot',
                                   comparison='same_history_geometry_with_saved_latent_lengths')
                    rows.append(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'new_model_calls': 0, 'new_rcaide_mda_calls': 0,
        'source_text_bindings_checked': checked, 'encoder_refits': 64, 'fitted_GP_count': 0,
        'optimizer_replay': False, 'rows': rows}, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    print({'encoder_refits': 64, 'new_model_calls': 0, 'rows': len(rows)})


if __name__ == '__main__':
    main()
