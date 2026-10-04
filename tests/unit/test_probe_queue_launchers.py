import importlib.util
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import write_json_atomic


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / 'scripts/phase1' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['run_regular_probe_pool_a', 'run_regular_fresh_rubrics'])
def test_archived_checkpoint_expansion_is_explicit_and_excludes_48(name):
    module = load(name)
    module.validate_steps([3, 6, 9])
    with pytest.raises(ValueError):
        module.validate_steps([13])
    module.validate_steps([13, 16, 32, 34, 40], include_archived=True)
    for bad in [[48], [47], [13, 13], []]:
        with pytest.raises(ValueError):
            module.validate_steps(bad, include_archived=True)


def test_fresh_queue_skips_unready_anchor_but_prioritizes_it_when_ready(tmp_path):
    module = load('run_regular_fresh_rubrics')
    def publish(step, pools):
        write_json_atomic(tmp_path / 'responses' / f'checkpoint-{step:06d}' / 'provenance.json',
                          {'selected_pools': pools}, immutable=False)
    assert module.choose_ready_step(tmp_path, [13, 16, 18]) is None
    publish(18, ['probe_A'])
    publish(13, ['probe_B'])
    assert module.choose_ready_step(tmp_path, [13, 16, 18]) == 18
    publish(13, ['probe_A', 'probe_B'])
    assert module.choose_ready_step(tmp_path, [13, 16, 18]) == 13


def test_predecessor_must_report_success(tmp_path):
    module = load('run_regular_probe_pool_a')
    status = tmp_path / 'status.json'
    write_json_atomic(status, {'state': 'failed'})
    with pytest.raises(RuntimeError, match='did not finish successfully'):
        module.wait_for_previous(2147483647, 'missing', status)
    write_json_atomic(status, {'state': 'complete'}, immutable=False)
    module.wait_for_previous(2147483647, 'missing', status)
