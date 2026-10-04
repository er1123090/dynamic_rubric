import importlib.util
from pathlib import Path

import pytest

from dynamic_rubric.providers.vllm_chat import VLLMChatError

spec = importlib.util.spec_from_file_location(
    'share_probe', Path(__file__).resolve().parents[2] / 'scripts/phase1/share_probe_column.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def test_prompt_groups_are_indivisible_and_cover_pool():
    rows = [{'prompt_id': p, 'response_id': f'{p}-{r}'} for p in ['a', 'b'] for r in range(16)]
    groups = target.split_responses(rows, {'trainer': ['a'], 'inference_a': ['b']})
    assert all(r['prompt_id'] == 'a' for r in groups['trainer'])
    assert len(groups['trainer']) == len(groups['inference_a']) == 16
    assert groups['trainer'] + groups['inference_a'] == rows


@pytest.mark.parametrize('assignment', [
    {'trainer': ['a'], 'inference_a': ['a', 'b']},
    {'trainer': ['a'], 'inference_a': ['c']},
    {'trainer': ['a', 'a'], 'inference_a': ['b']},
    {'trainer': ['a', 'b'], 'inference_a': []},
])
def test_invalid_assignment_rejected(assignment):
    with pytest.raises(ValueError):
        target.split_responses([{'prompt_id': 'a'}, {'prompt_id': 'b'}], assignment)


def test_only_transport_failures_are_retried():
    calls = []
    def transient():
        calls.append(1)
        if len(calls) < 2:
            raise VLLMChatError('transient')
        return 42
    assert target.retry_grading(transient, delay=0) == 42
    assert len(calls) == 2
    with pytest.raises(ValueError):
        target.retry_grading(lambda: (_ for _ in ()).throw(ValueError('identity')), delay=0)


def test_transport_retries_are_bounded():
    calls = []
    def fail():
        calls.append(1)
        raise VLLMChatError('persistent')
    with pytest.raises(VLLMChatError):
        target.retry_grading(fail, retries=2, delay=0)
    assert len(calls) == 3
