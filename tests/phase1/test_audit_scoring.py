import json
import pytest

from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool
from dynamic_rubric.providers.base import GenerationResult


class FakeGrader:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResult(
            text=json.dumps({"1": "PRESENT"}),
            requested_model="judge",
            returned_model="judge",
            request_id=f"r{self.calls}",
            created_at=None,
            retry_count=0,
        )


def test_score_pool_provenance_and_resume(tmp_path):
    rows = [
        {
            "response_id": "r1",
            "prompt_id": "p1",
            "prompt_messages": [{"role": "user", "content": "question"}],
            "text": "answer",
            "rollout_index": 0,
        }
    ]
    rubric = {"p1": [{"criterion_id": "c1", "text": "is correct", "weight": 1}]}
    grader = FakeGrader()
    cfg = AuditScoreConfig(domain="medicine", concurrency=1)
    first = score_pool(
        rows,
        rubric,
        evaluator_checkpoint="3",
        policy_checkpoint="3",
        config=cfg,
        grader=grader,
        cache_dir=tmp_path,
    )
    second = score_pool(
        rows,
        rubric,
        evaluator_checkpoint="3",
        policy_checkpoint="3",
        config=cfg,
        grader=grader,
        cache_dir=tmp_path,
    )
    assert first == second
    assert grader.calls == 1
    assert first[0]["pool"] == "probe_B"
    assert first[0]["fresh_or_stale"] == "fresh"
    assert first[0]["grades"] == [["c1", 1]]


def test_score_pool_rejects_missing_prompt_rubric():
    with pytest.raises(ValueError, match="non-empty"):
        score_pool(
            [{"response_id": "r", "prompt_id": "missing", "text": "a"}],
            {},
            evaluator_checkpoint="1",
            policy_checkpoint="0",
            config=AuditScoreConfig("m"),
            grader=FakeGrader(),
        )


def test_score_pool_marks_stale_evaluator():
    row = {
        "response_id": "r",
        "prompt_id": "p",
        "prompt_messages": [{"role": "user", "content": "q"}],
        "text": "a",
    }
    out = score_pool(
        [row],
        {"p": [{"criterion_id": "c", "text": "ok", "weight": 1}]},
        evaluator_checkpoint="1",
        policy_checkpoint="3",
        config=AuditScoreConfig("m"),
        grader=FakeGrader(),
    )
    assert out[0]["fresh_or_stale"] == "stale"
    assert out[0]["policy_step"] == 0
    assert out[0]["evaluator_step"] == 1


def test_score_pool_retains_endpoint_provenance(tmp_path):
    grader = FakeGrader()
    grader.request_provenance = lambda request: {
        "selected_base_url": "http://trainer:28007/v1",
        "provider_cache_path": str(tmp_path / "provider.json"),
    }
    out = score_pool(
        [{"response_id": "r", "prompt_id": "p", "text": "a"}],
        {"p": [{"criterion_id": "c", "text": "ok", "weight": 1}]},
        evaluator_checkpoint="1", policy_checkpoint="3",
        config=AuditScoreConfig("m"), grader=grader,
    )
    assert out[0]["judge"]["transport"]["selected_base_url"] == "http://trainer:28007/v1"


def test_vllm_transport_provenance_matches_immutable_cache_identity(tmp_path):
    from dynamic_rubric.providers.base import GenerationRequest
    from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter

    adapter = VLLMChatAdapter(["http://inference_b:8002", "http://trainer:28007"], "judge", tmp_path)
    request = GenerationRequest(prompt_id="p", messages=(), family="audit", seed=11)
    payload = adapter._payload(request)
    selected = adapter._select_base_url(request, payload)
    identity = adapter._identity(request, payload, selected)
    assert adapter.request_provenance(request) == {
        "selected_base_url": selected,
        "provider_cache_path": str(adapter._cache_path(identity).resolve()),
    }
