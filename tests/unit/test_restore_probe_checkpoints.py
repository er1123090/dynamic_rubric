import importlib.util
from pathlib import Path

import pytest

from dynamic_rubric.hashing import sha256_file

spec = importlib.util.spec_from_file_location('restore_probe', Path(__file__).resolve().parents[2] / 'scripts/phase1/restore_probe_checkpoints.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def test_restored_inventory_and_hash_are_required(tmp_path):
    p = tmp_path / 'actor/model.pt'
    p.parent.mkdir()
    p.write_bytes(b'original')
    rows = [{'path': 'actor/model.pt', 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}]
    target.validate_local(tmp_path, rows)
    p.write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        target.validate_local(tmp_path, rows)
    p.write_bytes(b'original')
    (tmp_path / 'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError, match='inventory'):
        target.validate_local(tmp_path, rows)


def test_symlinks_are_rejected(tmp_path):
    (tmp_path / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlinks'):
        target.validate_local(tmp_path, [])


@pytest.mark.parametrize('step', [-1, True, 45, 47, 48])
def test_no_latest_or_invalid_checkpoint_restoration(tmp_path, step):
    root = tmp_path / 'verl-run/checkpoints'
    root.mkdir(parents=True)
    (root / 'latest_checkpointed_iteration.txt').write_text('45')
    with pytest.raises(ValueError, match='historical checkpoints below latest'):
        target.restore(tmp_path, tmp_path, step, 'unused')


@pytest.mark.parametrize('step', [0, 3, 42])
def test_historical_restoration_requires_explicit_receipt(tmp_path, step):
    root = tmp_path / 'verl-run/checkpoints'
    root.mkdir(parents=True)
    (root / 'latest_checkpointed_iteration.txt').write_text('45')
    with pytest.raises(FileNotFoundError):
        target.restore(tmp_path, tmp_path, step, 'unused')
