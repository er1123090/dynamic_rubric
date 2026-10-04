#!/usr/bin/env python3
"""Exclusive Trainer GPU1 evaluation: real smoke first, then full; never stop others."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'scripts/phase1/evaluate_final_policies.py'
ORDER = ['static_base', 'static_final', 'online_base', 'online_final']


def require_free_gpu() -> None:
    result = subprocess.run(
        ['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True,
    )
    processes = subprocess.run(
        ['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True,
    )
    if processes.stdout.strip():
        raise RuntimeError(f'Trainer GPU1 has compute processes ({processes.stdout.strip()}); none stopped')
    if int(result.stdout.strip()) > 1024:
        raise RuntimeError(f'Trainer GPU1 occupied ({result.stdout.strip()} MiB); no process was stopped')


def stage(config: Path, action: str, model: str | None, limit: int | None) -> None:
    command = [sys.executable, str(RUNNER), '--config', str(config), '--stage', action]
    if model:
        command += ['--model', model]
    if limit:
        command += ['--limit', str(limit)]
    subprocess.run(command, cwd=ROOT, check=True)


def with_server(cfg: dict, name: str, log_dir: Path, task) -> None:
    require_free_gpu()
    judge = name == 'judge'
    spec = cfg['grading'] if judge else cfg['models'][name]
    port = 28132 if judge else 28131
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    model_path = Path(spec['local_path'])
    if not (model_path / 'config.json').is_file() or not list(model_path.glob('*.safetensors')):
        raise RuntimeError(f'Missing local inference export: {model_path}')
    command = [
        cfg['runtime']['vllm'], 'serve', str(model_path),
        '--served-model-name', spec['served_model'], '--tensor-parallel-size', '1',
        '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.90',
        '--max-model-len', '32768', '--max-num-seqs', '128',
        '--max-num-batched-tokens', '16384', '--enable-prefix-caching',
        '--generation-config', 'vllm', '--host', '127.0.0.1', '--port', str(port),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1')
    with (log_dir / f'{name}-{time.time_ns()}.log').open('ab') as log:
        require_free_gpu()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} vLLM exited before ready; inspect {log_dir}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                        ids = [m['id'] for m in json.load(response)['data']]
                    if ids != [spec['served_model']]:
                        raise RuntimeError(f'Wrong served model: {ids}')
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{name} server readiness timeout')
                    time.sleep(2)
            task()
        finally:
            # Only the process group created by this invocation is ever signalled.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
    deadline = time.monotonic() + 60
    while True:
        try:
            require_free_gpu()
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/final_policies_trainer_gpu1_20260910.yaml')
    parser.add_argument('--execute', action='store_true', help='Execute after preflight; default is read-only GPU preflight')
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text())
    if cfg['runtime']['gpu'] != 1 or cfg['runtime']['hostname'] != 'trainer':
        raise RuntimeError('This launcher is restricted to Trainer GPU1')
    require_free_gpu()
    if not args.execute:
        print('GPU1 preflight passed; no model launched')
        return
    output = (config.parent / cfg['output_root']).resolve()
    logs = output / 'launcher_logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (output / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for limit in (2, None):
            stage(config, 'prepare', None, limit)
            for name in ORDER:
                with_server(cfg, name, logs, lambda n=name: stage(config, 'generate', n, limit))
            with_server(cfg, 'judge', logs, lambda: stage(config, 'grade', None, limit))
            stage(config, 'summarize', None, limit)
        print('Completed smoke and full evaluation. All owned model servers stopped.')


if __name__ == '__main__':
    main()
