"""Summarize expected numerical domain trials without changing their exceptions."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys

_CURRENT = ContextVar('battery_trial_logging', default=None)


def log_domain_trial(error):
    state = _CURRENT.get()
    if state is None:
        print(f'[NASA battery] {type(error).__name__}: {error}', flush=True)
        return
    summary = state['summary']
    summary['total'] += 1
    message = str(error)
    category = message.split(':', 1)[0]
    counts = summary['counts_by_message_type']
    counts[category] = counts.get(category, 0) + 1
    summary['last_message'] = message
    event = {'trial_error': summary['total'], 'type': type(error).__name__, 'message': message}
    stream = state['stream']
    if stream is not None:
        try:
            stream.write(json.dumps(event, ensure_ascii=True) + '\n')
        except OSError as logging_error:
            summary['log_write_error'] = str(logging_error)
            state['stream'] = None
    if counts[category] == 1:
        print(f'[NASA battery] Rejected numerical trial: {message}; repeats summarized, '
              'not a final mission result', file=state['output'], flush=True)


@contextmanager
def summarize_domain_trials(diagnostics, artifact_dir=None):
    summary = {'scope': 'numerical_trial_errors_not_failed_aircraft_count',
               'total': 0, 'counts_by_message_type': {}}
    stream = None
    if artifact_dir is not None:
        log = Path(artifact_dir) / 'battery_domain_trials.jsonl'
        try:
            stream = log.open('x', encoding='utf-8', buffering=1)
            summary['log_file'] = str(log)
        except OSError as error:
            summary['log_open_error'] = str(error)
    state = {'summary': summary, 'stream': stream, 'output': sys.stdout}
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        _CURRENT.reset(token)
        if stream is not None:
            try:
                stream.close()
            except OSError as error:
                summary['log_close_error'] = str(error)
        diagnostics['battery_domain_trial_log'] = summary
        if summary['total']:
            print(f'[NASA battery] Numerical trial rejections: {summary["total"]}; '
                  f'details: {summary.get("log_file", "diagnostics metadata")}. '
                  'See final candidate status for mission success/failure.',
                  file=state['output'], flush=True)
