"""Restore pinned historical policy parameters for offline audits, never training."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import time

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.phase1.audit_policy import export_checkpoint, load_run_contract
from dynamic_rubric.phase1.audit_checkpoint_identity import inspect_scoring_checkpoint
from dynamic_rubric.training.checkpoint_archive import _validate_files

ALLOWED_STEPS = (13, 16, 32, 34, 40)


def validate_local(root: Path, records: list[dict]) -> None:
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('checkpoint symlinks are forbidden')
    paths = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if paths != {r['path'] for r in records}:
        raise ValueError('restored checkpoint file inventory differs from archive')
    for row in records:
        p = root / row['path']
        if p.stat().st_size != row['bytes'] or sha256_file(p) != row['sha256']:
            raise ValueError(f'restored checkpoint checksum mismatch: {row["path"]}')


def restore(run: Path, audit: Path, step: int, hf_cli: str) -> None:
    latest = int((run / 'verl-run/checkpoints/latest_checkpointed_iteration.txt').read_text().strip())
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < latest:
        raise ValueError('only explicitly archived historical checkpoints below latest may be restored')
    receipt = read_json(run / 'verl-run/checkpoint_archives' / f'global_step_{step}.json')
    if receipt.get('run_id') != run.name or receipt.get('checkpoint_step') != step:
        raise ValueError('archive receipt is not bound to the requested run and step')
    records = _validate_files(receipt['files'])
    contract = load_run_contract(run)
    inspect_scoring_checkpoint(contract, step)
    checkpoint_root = run / 'verl-run/checkpoints'
    target = checkpoint_root / f'global_step_{step}'
    if checkpoint_root.is_symlink() or target.is_symlink():
        raise ValueError('checkpoint targets must be canonical directories')
    if not target.exists():
        download = audit / 'restored_archives' / f'global_step_{step}'
        download.mkdir(parents=True, exist_ok=True)
        subprocess.run([hf_cli, 'download', receipt['repo_id'],
                        *[r['remote_path'] for r in records],
                        '--revision', receipt['revision'], '--local-dir', str(download),
                        '--max-workers', '2', '--quiet'], check=True)
        staged = download / 'original_checkpoint'
        validate_local(staged, records)
        if target.exists():
            raise RuntimeError('checkpoint target appeared during restore; refuse overwrite')
        staged.rename(target)
    validate_local(target, records)
    write_json_atomic(audit / 'logs' / f'restore-checkpoint-{step}.json', {
        'state': 'verified', 'checkpoint_step': step, 'repo_id': receipt['repo_id'],
        'revision': receipt['revision'], 'actor_parameter_hash': receipt['actor_parameter_hash'],
        'restored_at': time.time(), 'optimizer_restored': False,
    }, immutable=False)
    project = Path(__file__).resolve().parents[2]
    verl = project / 'environment/upstream/verl'
    export_checkpoint(contract, step=step, export_root=audit / 'exports',
                      merger_python=str(verl / '.venv-runtime/bin/python'), verl_root=verl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--steps', nargs='+', type=int, default=list(ALLOWED_STEPS))
    parser.add_argument('--hf-cli', default="hf")
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(step < 0 for step in args.steps):
        parser.error('steps must be unique nonnegative archived checkpoint numbers')
    root = args.artifact_root.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    status = root / 'logs/restore_matrix_status.json'
    completed = []
    with (root / 'logs/restore_matrix.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for step in args.steps:
            write_json_atomic(status, {'state': 'restoring_and_exporting', 'checkpoint': step,
                                      'completed_checkpoints': completed}, immutable=False)
            try:
                restore(args.run_dir.resolve(), root, step, args.hf_cli)
            except Exception as error:
                write_json_atomic(status, {'state': 'failed', 'checkpoint': step,
                                          'error': repr(error), 'completed_checkpoints': completed}, immutable=False)
                raise
            completed.append(step)
        write_json_atomic(status, {'state': 'complete', 'completed_checkpoints': completed}, immutable=False)


if __name__ == '__main__':
    main()
