import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import read_json
from dynamic_rubric.providers.vllm_chat import VLLMChatError

spec = importlib.util.spec_from_file_location('scorer_cli', Path(__file__).resolve().parents[2] / 'scripts/phase1/score_regular_probe_adjacent.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def args(root):
    return SimpleNamespace(run_dir=root, artifact_root=root, output_root=root,
                           steps=[], judge_urls=['http://judge'], concurrency=32,
                           wait_timeout=0, cell_plan=None, bounded_grading_whitespace=True,
                           max_client_restarts=2, restart_delay=0)


def test_transient_provider_failure_restarts_with_same_configuration(tmp_path, monkeypatch):
    calls = []
    def score(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise VLLMChatError('temporary transport failure')
        return {'state': 'complete'}
    monkeypatch.setattr(target, 'score_regular_adjacent', score)
    assert target.run_with_restarts(args(tmp_path)) == {'state': 'complete'}
    assert calls[0] == calls[1]


@pytest.mark.parametrize('error,expected_calls', [(VLLMChatError('persistent'), 3), (ValueError('provenance'), 1)])
def test_retry_is_bounded_and_never_masks_provenance_failure(tmp_path, monkeypatch, error, expected_calls):
    calls = []
    def score(**kwargs):
        calls.append(kwargs)
        raise error
    monkeypatch.setattr(target, 'score_regular_adjacent', score)
    with pytest.raises(type(error)):
        target.run_with_restarts(args(tmp_path))
    assert len(calls) == expected_calls
    assert read_json(tmp_path / 'status.json')['state'] == 'failed'
