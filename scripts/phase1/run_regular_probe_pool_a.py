"""Trainer GPU1: independent probe pools, optionally sharing with the extractor."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.audit_policy import generate_probe_pools, load_run_contract


def ready(process, url, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Server exited: {process.args}')
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(url)


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    # These groups were created by this launcher only, never a shared training/server group.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def validate_steps(steps, include_archived=False):
    allowed = set(range(0, 46, 3))
    if include_archived:
        allowed.update((13, 16, 32, 34, 40))
    if not steps or len(set(steps)) != len(steps) or not set(steps) <= allowed:
        raise ValueError('invalid or duplicate probe checkpoint steps')


def wait_for_previous(pid, starttime, status_path):
    while True:
        stat = Path(f'/proc/{pid}/stat')
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            break
        if fields[19] != str(starttime) or fields[0] in ('Z', 'X'):
            break
        time.sleep(10)
    if read_json(status_path).get('state') != 'complete':
        raise RuntimeError('previous pool launcher did not finish successfully; refuse overlapping generation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--steps', nargs='+', type=int, default=list(range(0, 46, 3)))
    parser.add_argument('--extractor-base-url')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--include-pool-b', action='store_true')
    parser.add_argument('--policy-python', default=str(Path(__file__).resolve().parents[2] / '.venvs/judge/bin/python'))
    parser.add_argument('--include-archived-checkpoints', action='store_true')
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--wait-for-starttime')
    args = parser.parse_args()
    contract = load_run_contract(args.run_dir)
    root = args.output_root.resolve()
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    validate_steps(args.steps, args.include_archived_checkpoints)
    if (args.wait_for_pid is None) != (args.wait_for_starttime is None):
        parser.error('--wait-for-pid and --wait-for-starttime must be supplied together')
    if args.wait_for_pid is not None:
        wait_for_previous(args.wait_for_pid, args.wait_for_starttime, logs / 'pool_a_status.json')
    stop_receipt = read_json(contract.run_dir / 'logs/stop-after-step47.json')
    for pid, identity in stop_receipt['pids'].items():
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if fields[19] == identity[1] and fields[0] not in ('Z', 'X'):
                raise RuntimeError(f'Training process {pid} is still alive; refuse probe launch')
    memory = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if args.extractor_base_url:
        with urllib.request.urlopen(args.extractor_base_url.rstrip('/') + '/v1/models', timeout=5) as response:
            models = json.loads(response.read())
        assert {row['id'] for row in models['data']} == {'openai/gpt-oss-120b'}
        assert 0 < args.gpu_memory_utilization <= 0.20
    elif memory > 512:
        raise RuntimeError(f'Trainer GPU1 has {memory} MiB in use; training shutdown not verified')
    tokenizer = Path(__file__).resolve().parents[2] / 'models/Qwen3-4B-Instruct-2507'
    assert tokenizer.is_dir()
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'PYTHONUNBUFFERED': '1'}
    timings_path = logs / 'pool_a_timings.json'
    timings = read_json(timings_path) if timings_path.is_file() else []
    for step in args.steps:
        provenance = root / 'responses' / f'checkpoint-{step:06d}' / 'provenance.json'
        required_pools = ('probe_B',) if step == 0 and args.include_pool_b else (('probe_A', 'probe_B') if args.include_pool_b else ('probe_A',))
        if provenance.is_file() and set(required_pools) <= set(read_json(provenance).get('selected_pools', ['probe_A', 'probe_B'])):
            result = generate_probe_pools(contract, step=step, output_root=root, base_url='http://127.0.0.1:28006', pools=required_pools)
            assert result['reused']
            timings.append({'checkpoint': step, 'reused': True})
            print({'checkpoint': step, 'reused': True}, flush=True)
            continue
        export = root / 'exports' / f'global_step_{step}'
        deadline = time.monotonic() + 86400
        while not (export / 'audit_export_manifest.json').is_file():
            restore_status = root / 'logs/restore_matrix_status.json'
            if restore_status.is_file() and read_json(restore_status).get('state') == 'failed':
                raise RuntimeError('checkpoint restoration failed; inspect restore_matrix_status.json')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Export not ready for checkpoint {step}')
            time.sleep(5)
        manifest = read_json(export / 'audit_export_manifest.json')
        started = time.time()
        server = proxy = None
        write_json_atomic(root / 'logs/pool_a_status.json', {'state': 'running', 'checkpoint': step, 'started_at': started,
                          'completed_checkpoints': [item['checkpoint'] for item in timings]}, immutable=False)
        try:
            with (logs / f'policy-vllm-{step}.log').open('ab') as server_log, (logs / f'policy-proxy-{step}.log').open('ab') as proxy_log:
                server = subprocess.Popen([args.policy_python, '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', str(export), '--tokenizer', str(tokenizer), '--served-model-name', contract.model,
                    '--host', '127.0.0.1', '--port', '28005', '--dtype', 'bfloat16', '--max-model-len', '8192',
                    '--generation-config', 'vllm', '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--enable-prefix-caching',
                    '--max-num-batched-tokens', '16384', '--max-num-seqs', '128'],
                    env=environment, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                ready(server, 'http://127.0.0.1:28005/v1/models')
                proxy = subprocess.Popen([sys.executable, '-m', 'dynamic_rubric.services.vllm_policy_proxy',
                    '--upstream', 'http://127.0.0.1:28005', '--model-path', str(export), '--served-model', contract.model,
                    '--model-revision', contract.model_revision, '--tokenizer-revision', contract.tokenizer_revision,
                    '--checkpoint-hash', manifest['source_model_sha256'], '--port', '28006', '--timeout-seconds', '900'],
                    env=environment, stdout=proxy_log, stderr=subprocess.STDOUT, start_new_session=True)
                ready(proxy, 'http://127.0.0.1:28006/health', timeout=60)
                generation_started = time.time()
                if step != 0 or not args.include_pool_b:
                    result = generate_probe_pools(contract, step=step, output_root=root, base_url='http://127.0.0.1:28006',
                                                  pools=('probe_A',), concurrency=32)
                    print({'checkpoint': step, 'pool_A_ready_at': time.time(), 'result': result}, flush=True)
                if args.include_pool_b:
                    result = generate_probe_pools(contract, step=step, output_root=root, base_url='http://127.0.0.1:28006',
                                                  pools=('probe_B',), concurrency=32)
                timing = {'checkpoint': step, 'started_at': started, 'generation_started_at': generation_started,
                          'finished_at': time.time(), 'result': result}
                timings.append(timing)
                print(timing, flush=True)
                write_json_atomic(logs / 'pool_a_timings.json', timings, immutable=False)
        except Exception as error:
            write_json_atomic(logs / 'pool_a_status.json', {'state': 'failed', 'checkpoint': step,
                              'error': repr(error), 'completed_checkpoints': [item['checkpoint'] for item in timings]}, immutable=False)
            raise
        finally:
            stop_owned(proxy)
            stop_owned(server)
    write_json_atomic(logs / 'pool_a_status.json', {'state': 'complete', 'completed_checkpoints': args.steps,
                      'finished_at': time.time()}, immutable=False)


if __name__ == '__main__':
    main()
