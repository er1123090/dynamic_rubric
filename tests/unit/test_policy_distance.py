from __future__ import annotations

from dynamic_rubric.training.policy_distance import (
    PolicyDistanceEnricher,
    summarize_policy_distance,
)


class _Embedding:
    identity = {"model": "test"}

    def embed(self, texts):
        return [
            [1.0, 0.0] if "reference" in text or "same" in text else [0.0, 1.0]
            for text in texts
        ]


def _references() -> list[dict[str, str]]:
    rows = []
    for family, count in (("reference_discovery", 8), ("reference_validation", 4)):
        for index in range(count):
            rows.append(
                {
                    "prompt_id": "p",
                    "family": family,
                    "response_id": f"{family}-{index}",
                    "response_text": f"reference {index}",
                }
            )
    return rows


def test_policy_distance_enriches_kl_embedding_length_and_style() -> None:
    enricher = PolicyDistanceEnricher.from_references(_Embedding(), _references())
    rows = enricher.enrich(
        [
            {
                "prompt_id": "p",
                "split": "pilot_probe",
                "policy_step": 1,
                "family": "trajectory_discovery",
                "response_id": "current",
                "response_text": "# same\n- item",
                "kl_from_pi0": 0.25,
                "static_proxy_reward": 0.75,
            }
        ]
    )

    assert rows[0]["response_embedding_distance"] == 0.0
    assert rows[0]["response_length_words"] == 2
    assert rows[0]["style_markdown_heading"] is True
    assert rows[0]["style_bullet_list"] is True
    summary = summarize_policy_distance(rows)[0]
    assert summary["mean_kl_from_pi0"] == 0.25
    assert summary["mean_static_proxy_reward"] == 0.75
    assert summary["markdown_heading_rate"] == 1.0
