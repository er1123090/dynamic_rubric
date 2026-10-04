"""Execute the trainer's actual gate AST without loading GPUs or Ray workers."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "environment/upstream/verl/verl/trainer/ppo/ray_trainer.py"


def gate():
    tree = ast.parse(TRAINER.read_text())
    nodes = list(ast.walk(tree))
    branches = [node for node in nodes if isinstance(node, ast.If)
                and ast.unparse(node.test) == "prefetched_logprobs is None or verify_prefetch"]
    assert len(branches) == 2, "trainer must gate both old and reference log-probs"
    verify = next(node for node in nodes if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == "verify_prefetch")
    finalize = next(node for node in nodes if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "verify_prefetch"
                    and any(isinstance(part, ast.Assign) and ast.unparse(part.targets[0]) == "self._online_prefetch_verified"
                            for part in node.body))
    validate = next(node for node in nodes if isinstance(node, ast.If)
                    and any(isinstance(part, ast.Expr)
                            and ast.unparse(part.value) == "prefetched_logprobs.validate_inputs(batch)"
                            for part in node.body))
    function = ast.parse("def run(self, batch, prefetched_logprobs, metrics): pass").body[0]
    function.body = copy.deepcopy([validate, verify]) + ast.parse(
        "prefetch_exact = True\ncalculate_entropy = False"
    ).body + copy.deepcopy(branches + [finalize]) + ast.parse(
        "return old_log_prob, ref_log_prob"
    ).body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["run"]


class Trainer:
    def __init__(self, verified=False):
        self._online_prefetch_verified = verified
        self._online_prefetch_disabled = False
        self.calls = []
        self.old = object()
        self.ref = object()

    def _compute_old_log_prob(self, batch, *, calculate_entropy):
        assert calculate_entropy is False
        self.calls.append("old")
        return self.old, 0.1

    def _compute_ref_log_prob(self, batch):
        self.calls.append("ref")
        return self.ref


class Prefetch:
    def __init__(self, old_exact=True, ref_exact=True, invalid=False):
        self.old, self.ref, self.old_mfu = object(), object(), 0.2
        self.exact = {"old": old_exact, "ref": ref_exact}
        self.invalid = invalid
        self.validations = 0

    def validate_inputs(self, batch):
        self.validations += 1
        if self.invalid:
            raise RuntimeError("row alignment changed")

    def matches(self, actual, kind):
        return self.exact[kind]


def test_disabled_path_calls_original_old_then_reference():
    trainer = Trainer()
    trainer.global_steps = 33
    assert gate()(trainer, None, None, {}) == (trainer.old, trainer.ref)
    assert trainer.calls == ["old", "ref"]


@pytest.mark.parametrize(("old_exact", "ref_exact"), [(True, True), (False, True), (True, False)])
def test_first_batch_always_uses_serial_outputs_and_exact_gate(old_exact, ref_exact):
    trainer = Trainer()
    trainer.global_steps = 33
    prefetched = Prefetch(old_exact, ref_exact)
    assert gate()(trainer, None, prefetched, {}) == (trainer.old, trainer.ref)
    assert trainer.calls == ["old", "ref"]
    assert prefetched.validations == 1
    assert trainer._online_prefetch_verified is (old_exact and ref_exact)
    assert trainer._online_prefetch_disabled is not (old_exact and ref_exact)


def test_verified_subsequent_batch_reuses_only_aligned_outputs():
    trainer = Trainer(verified=True)
    prefetched = Prefetch()
    assert gate()(trainer, None, prefetched, {}) == (prefetched.old, prefetched.ref)
    assert trainer.calls == []
    assert prefetched.validations == 1


def test_alignment_failure_blocks_all_downstream_calculation():
    trainer = Trainer(verified=True)
    with pytest.raises(RuntimeError, match="alignment"):
        gate()(trainer, None, Prefetch(invalid=True), {})
    assert trainer.calls == []


def test_snapshot_preprocessing_exactly_matches_original_trainer():
    helper = (ROOT / "src/dynamic_rubric/training/logprob_prefetch.py").read_text()
    start = 'if "response_mask" not in batch.batch.keys():'
    end = 'batch.meta_info["images_seqlens"] = images_seqlens_all'

    def extract(source):
        first = source.index(start)
        # Retain indentation for dedent, including the first line.
        first = source.rfind("\n", 0, first) + 1
        last = source.index(end, first) + len(end)
        return ast.parse(textwrap.dedent(source[first:last]).replace("self.", "trainer."))

    assert ast.dump(extract(TRAINER.read_text())) == ast.dump(extract(helper))


def test_launcher_defaults_off_and_records_execution_only_flag():
    launcher = (ROOT / "scripts/phase1/run_online_full.sh").read_text()
    assert "ONLINE_LOGPROB_PREFETCH=${ONLINE_LOGPROB_PREFETCH:-false}" in launcher
    assert '++reward.online_step_hook.prefetch_logprobs="${ONLINE_LOGPROB_PREFETCH}"' in launcher
    spec = (ROOT / "src/dynamic_rubric/phase1/full_run.py").read_text()
    assert '"execution_optimizations"' in spec


@pytest.mark.parametrize(("count", "expected"), [(1536, True), (960, False)])
def test_partial_batch_never_uses_unvalidated_prefetch_shape(count, expected):
    tree = ast.parse(TRAINER.read_text())
    expression = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "overlap_enabled")
    trainer = SimpleNamespace(config=SimpleNamespace(
        data=SimpleNamespace(train_batch_size=96),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=16)),
    ))
    actual = eval(compile(ast.Expression(expression), str(TRAINER), "eval"), {
        "self": trainer, "batch": range(count), "hook_config": {"prefetch_logprobs": True}
    })
    assert actual is expected
