"""Checkpoint identity for scoring saved responses, never for model loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..artifacts import read_json, validate_artifact_record
from ..training.checkpoint_archive import verify_public_archive
from .audit_policy import AuditPolicyError, CheckpointIdentity, inspect_checkpoint


@lru_cache(maxsize=128)
def _verified_archive(serialized_receipt: str) -> str:
    # The complete receipt is the cache key; edits force re-verification.
    return verify_public_archive(json.loads(serialized_receipt))


def inspect_scoring_checkpoint(contract, step: int) -> CheckpointIdentity:
    """Use pinned, verified archival bytes to validate saved scoring inputs.

    Training, export, rollout generation and KL retain their strict local-file
    requirements. Existing response/rubric hash checks still bind this identity.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise AuditPolicyError("invalid scoring checkpoint step")
    receipt_path = contract.run_dir / 'verl-run/checkpoint_archives' / f'global_step_{step}.json'
    if not receipt_path.is_file():
        try:
            return inspect_checkpoint(contract, step)
        except (FileNotFoundError, AuditPolicyError):
            # An archival publisher may have committed its receipt and removed
            # the original between the existence test and the local inspection.
            if not receipt_path.is_file():
                raise
    receipt = read_json(receipt_path)
    if receipt.get('run_id') != contract.run_id or receipt.get('checkpoint_step') != step:
        raise AuditPolicyError('archive receipt run/checkpoint identity mismatch')
    actor_hash = _verified_archive(json.dumps(receipt, sort_keys=True))
    source = next(item for item in receipt['files']
                  if item['path'] == 'actor/model_world_size_1_rank_0.pt')
    if step:
        commit = read_json(contract.run_dir / 'verl-run/online_steps' / f'step-{step:06d}/commit.json')
        if commit.get('artifacts', {}).get('actor_parameter_hash') != actor_hash:
            raise AuditPolicyError('archive differs from committed actor parameters')
    else:
        binding = receipt.get('initial_policy_provenance')
        if not isinstance(binding, dict):
            raise AuditPolicyError('initial checkpoint archive lacks initial-policy binding')
        path = Path(str(binding.get('path', ''))).resolve()
        if (not path.is_relative_to(contract.run_dir.parent.resolve())
                or path.parts[-3:] != ('responses', 'checkpoint-000000', 'provenance.json')):
            raise AuditPolicyError('initial-policy binding path is outside this run workspace')
        validate_artifact_record(binding)
        proof = read_json(path)
        expected = {'run_id': contract.run_id, 'global_step': 0,
                    'checkpoint_hash': source['sha256'],
                    'config_sha256': contract.config_sha256,
                    'launch_spec_sha256': contract.launch_spec_sha256,
                    'probe_manifest_sha256': contract.probe_manifest_sha256,
                    'train_sha256': contract.train_sha256}
        if any(proof.get(key) != value for key, value in expected.items()):
            raise AuditPolicyError('initial archive differs from saved initial-policy provenance')
    actor = contract.run_dir / 'verl-run/checkpoints' / f'global_step_{step}/actor'
    return CheckpointIdentity(step=step, actor_dir=actor,
                              source_model=actor / 'model_world_size_1_rank_0.pt',
                              source_model_sha256=source['sha256'],
                              source_model_bytes=source['bytes'])
