import importlib.util
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import artifact_record, write_json_atomic

spec = importlib.util.spec_from_file_location('trainer_transition', Path(__file__).resolve().parents[2] / 'scripts/phase1/transition_trainer_to_judge.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def status(root, name, value):
    write_json_atomic(root / 'logs' / name, value, immutable=False)


def test_does_not_unlock_before_both_producers_finish(tmp_path):
    status(tmp_path, 'fresh_queue_status.json', {'state': 'running'})
    status(tmp_path, 'pool_a_status.json', {'state': 'complete'})
    assert not target.generation_complete(tmp_path)
    status(tmp_path, 'fresh_queue_status.json', {'state': 'complete', 'completed_checkpoints': [40]})
    with pytest.raises(RuntimeError, match='omits'):
        target.generation_complete(tmp_path)


def test_checks_all_pool_counts_and_rubric_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(target, 'REQUIRED_FRESH', {3})
    status(tmp_path, 'fresh_queue_status.json', {'state': 'complete', 'completed_checkpoints': [3]})
    status(tmp_path, 'pool_a_status.json', {'state': 'complete'})
    for step in [0, 3]:
        write_json_atomic(tmp_path / 'responses' / f'checkpoint-{step:06d}' / 'provenance.json',
                          {'global_step': step, 'pool_counts': {'probe_A': 800, 'probe_B': 1600}, 'artifacts': []})
    rubric = tmp_path / 'rubrics/checkpoint-000003/fresh_rubrics.jsonl'
    rubric.parent.mkdir(parents=True)
    rubric.write_text('{}\n')
    write_json_atomic(rubric.parent / 'status.json', {'state': 'complete', 'prompt_count': 100,
                      'fresh_rubrics_sha256': artifact_record(rubric)['sha256']})
    assert target.generation_complete(tmp_path)
    rubric.write_text('changed')
    with pytest.raises(RuntimeError, match='rubric inventory'):
        target.generation_complete(tmp_path)


def test_never_signals_reused_pid(monkeypatch):
    monkeypatch.setattr(target, 'process_identity', lambda pid: {'starttime': 'new', 'state': 'R'})
    signals = []
    monkeypatch.setattr(target.os, 'kill', lambda *args: signals.append(args))
    target.stop_exact({'pid': 5, 'starttime': 'old'})
    assert not signals


def test_replica_version_and_context_must_match():
    primary = {'version': {'version': '0.19.1'}, 'model': {'id': target.MODEL, 'max_model_len': 32768}}
    target.verify_replica(primary, primary)
    with pytest.raises(RuntimeError, match='vLLM'):
        target.verify_replica(primary, {**primary, 'version': {'version': '0.20.1'}})
    with pytest.raises(RuntimeError, match='max_model_len'):
        target.verify_replica(primary, {**primary, 'model': {**primary['model'], 'max_model_len': 8192}})


def test_scorer_uses_distinct_output_and_one_endpoint(tmp_path):
    command = target.scorer_command(tmp_path, tmp_path, tmp_path / 'scores-trainer', tmp_path / 'trainer-plan.json', 'http://127.0.0.1:28012')
    assert command[command.index('--output-root') + 1].endswith('scores-trainer')
    assert command[command.index('--judge-urls') + 1] == 'http://127.0.0.1:28012'
    assert command[command.index('--concurrency') + 1] == '32'
