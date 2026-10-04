import json
from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import artifact_record, write_json_atomic
from dynamic_rubric.phase1 import audit_checkpoint_identity as target
from dynamic_rubric.phase1.audit_policy import AuditPolicyError, inspect_checkpoint


def setup(tmp_path, monkeypatch, step=3):
    contract = SimpleNamespace(run_dir=tmp_path/'run', run_id='run',
                               config_sha256='config', launch_spec_sha256='launch',
                               probe_manifest_sha256='probe', train_sha256='train')
    receipt = {'run_id': 'run', 'checkpoint_step': step, 'actor_parameter_hash': 'tree',
               'files': [{'path': 'actor/model_world_size_1_rank_0.pt',
                          'sha256': 'file-sha', 'bytes': 123}]}
    path = contract.run_dir/'verl-run/checkpoint_archives'/f'global_step_{step}.json'
    write_json_atomic(path, receipt)
    if step:
        write_json_atomic(contract.run_dir/'verl-run/online_steps'/f'step-{step:06d}/commit.json',
                          {'artifacts': {'actor_parameter_hash': 'tree'}})
    monkeypatch.setattr(target, '_verified_archive', lambda value: json.loads(value)['actor_parameter_hash'])
    return contract, receipt, path


def test_missing_local_uses_file_hash_not_actor_tree_hash(tmp_path, monkeypatch):
    contract, _, _ = setup(tmp_path, monkeypatch)
    identity = target.inspect_scoring_checkpoint(contract, 3)
    assert identity.source_model_sha256 == 'file-sha'
    assert identity.source_model_bytes == 123
    assert not identity.source_model.exists()
    with pytest.raises(AuditPolicyError):
        inspect_checkpoint(contract, 3)  # model-loading paths remain strict


@pytest.mark.parametrize('field,value', [('run_id', 'other'), ('checkpoint_step', 6),
                                       ('actor_parameter_hash', 'wrong')])
def test_archive_mismatch_is_not_accepted(tmp_path, monkeypatch, field, value):
    contract, receipt, path = setup(tmp_path, monkeypatch)
    receipt[field] = value
    write_json_atomic(path, receipt, immutable=False)
    with pytest.raises(AuditPolicyError):
        target.inspect_scoring_checkpoint(contract, 3)


def test_initial_requires_hash_bound_provenance(tmp_path, monkeypatch):
    contract, receipt, path = setup(tmp_path, monkeypatch, step=0)
    with pytest.raises(AuditPolicyError, match='initial-policy binding'):
        target.inspect_scoring_checkpoint(contract, 0)
    proof = tmp_path/'audit/responses/checkpoint-000000/provenance.json'
    values = {'run_id': 'run', 'global_step': 0, 'checkpoint_hash': 'file-sha',
              'config_sha256': 'config', 'launch_spec_sha256': 'launch',
              'probe_manifest_sha256': 'probe', 'train_sha256': 'train'}
    write_json_atomic(proof, values)
    receipt['initial_policy_provenance'] = artifact_record(proof)
    write_json_atomic(path, receipt, immutable=False)
    assert target.inspect_scoring_checkpoint(contract, 0).source_model_sha256 == 'file-sha'
    values['checkpoint_hash'] = 'changed'
    write_json_atomic(proof, values, immutable=False)
    receipt['initial_policy_provenance'] = artifact_record(proof)
    write_json_atomic(path, receipt, immutable=False)
    with pytest.raises(AuditPolicyError, match='saved initial-policy'):
        target.inspect_scoring_checkpoint(contract, 0)


def test_remote_failure_propagates(tmp_path, monkeypatch):
    contract, _, _ = setup(tmp_path, monkeypatch)
    def fail(_):
        raise RuntimeError('remote checksum changed')
    monkeypatch.setattr(target, '_verified_archive', fail)
    with pytest.raises(RuntimeError, match='checksum'):
        target.inspect_scoring_checkpoint(contract, 3)


def test_no_archive_preserves_original_inspection(tmp_path, monkeypatch):
    contract = SimpleNamespace(run_dir=tmp_path)
    sentinel = object()
    monkeypatch.setattr(target, 'inspect_checkpoint', lambda c, step: sentinel)
    assert target.inspect_scoring_checkpoint(contract, 3) is sentinel
