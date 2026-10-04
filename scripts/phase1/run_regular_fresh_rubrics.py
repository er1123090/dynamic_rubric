"""Trainer gpt-oss queue: fresh rubrics only, as independent Pool A becomes ready."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from dynamic_rubric.artifacts import read_json, validate_artifact_record, write_json_atomic
from dynamic_rubric.phase1.audit_policy import load_run_contract


def validate_steps(steps, include_archived=False):
    allowed = set(range(3, 46, 3))
    if include_archived:
        allowed.update((13, 16, 32, 34, 40))
    if not steps or len(set(steps)) != len(steps) or not set(steps) <= allowed:
        raise ValueError('invalid or duplicate fresh-rubric checkpoint steps')


def choose_ready_step(root, pending):
    for step in pending:
        path = root / 'responses' / f'checkpoint-{step:06d}' / 'provenance.json'
        if path.is_file() and 'probe_A' in read_json(path).get('selected_pools', ['probe_A', 'probe_B']):
            return step
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=list(range(3, 46, 3)))
    parser.add_argument('--endpoint', default='http://127.0.0.1:28011')
    parser.add_argument('--prompt-workers', type=int, default=8)
    parser.add_argument('--extractor-concurrency', type=int, default=8)
    parser.add_argument('--max-in-flight', type=int, default=32)
    parser.add_argument('--include-archived-checkpoints', action='store_true')
    parser.add_argument('--ready-queue', action='store_true')
    args = parser.parse_args()
    contract = load_run_contract(args.run_dir)
    root = args.output_root.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    launch = read_json(contract.launch_spec_path)
    controls = launch['control_cache']['path']
    validate_steps(args.steps, args.include_archived_checkpoints)
    timings_path = root / 'logs/fresh_queue_timings.json'
    timings = read_json(timings_path) if timings_path.is_file() else []
    pending = list(args.steps)
    while pending:
        step = pending[0]
        if args.ready_queue:
            deadline = time.monotonic() + 86400
            while (step := choose_ready_step(root, pending)) is None:
                write_json_atomic(root / 'logs/fresh_queue_status.json', {
                    'state': 'waiting_for_any_pool_A', 'pending_checkpoints': pending,
                    'completed_checkpoints': [item['checkpoint'] for item in timings],
                }, immutable=False)
                if time.monotonic() > deadline:
                    raise TimeoutError('No verified Pool A became ready within 24 hours')
                time.sleep(5)
        responses = root / 'responses' / f'checkpoint-{step:06d}'
        status = root / 'logs/fresh_queue_status.json'
        write_json_atomic(status, {'state': 'waiting_for_pool_A', 'checkpoint': step,
                          'completed_checkpoints': [item['checkpoint'] for item in timings]}, immutable=False)
        deadline = time.monotonic() + 7200
        while not (responses / 'provenance.json').is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f'No verified Pool A for checkpoint {step}')
            time.sleep(5)
        provenance = read_json(responses / 'provenance.json')
        for artifact in provenance['artifacts']:
            validate_artifact_record(artifact)
        assert provenance['config_sha256'] == contract.config_sha256
        assert provenance['probe_manifest_sha256'] == contract.probe_manifest_sha256
        assert provenance['global_step'] == step
        started = time.time()
        write_json_atomic(status, {'state': 'extracting_and_deduplicating', 'checkpoint': step,
                          'started_at': started, 'completed_checkpoints': [item['checkpoint'] for item in timings]}, immutable=False)
        command = [sys.executable, 'scripts/phase1/build_fixed_train_fresh_rubrics.py',
                   '--run-dir', str(contract.run_dir), '--run-id', contract.run_id, '--train', str(contract.train_path),
                   '--probe-manifest', str(contract.probe_manifest_path), '--pool-a', str(responses / 'probe_A.jsonl'),
                   '--pi0-manifest', controls, '--checkpoint-step', str(step),
                   '--checkpoint-hash', provenance['checkpoint_hash'], '--output-root', str(root / 'rubrics'), '--endpoint', args.endpoint,
                   '--seed', str(contract.seed), '--prompt-workers', str(args.prompt_workers),
                   '--extractor-concurrency', str(args.extractor_concurrency),
                   '--max-in-flight', str(args.max_in_flight), '--timeout', '900']
        with (root / 'logs' / f'fresh-rubrics-{step}.log').open('ab') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        summary = read_json(root / 'rubrics' / f'checkpoint-{step:06d}' / 'status.json')
        assert summary['state'] == 'complete' and summary['prompt_count'] == 100
        timing = {'checkpoint': step, 'started_at': started, 'finished_at': time.time(), 'prompt_count': 100}
        if step not in {item['checkpoint'] for item in timings}:
            timings.append(timing)
        print(timing, flush=True)
        write_json_atomic(root / 'logs/fresh_queue_timings.json', timings, immutable=False)
        pending.remove(step)
    write_json_atomic(root / 'logs/fresh_queue_status.json', {'state': 'complete', 'completed_checkpoints': args.steps,
                      'finished_at': time.time()}, immutable=False)


if __name__ == '__main__':
    main()
