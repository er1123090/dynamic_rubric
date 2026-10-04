from __future__ import annotations

import json

from dynamic_rubric.judge_prompts import (
    PAPER_JUDGE_PROMPT_VERSION,
    PAPER_JUDGE_SYSTEM_PROMPT,
    qwen_criterion_prompt,
)
from dynamic_rubric.minimum_gold import _request
from dynamic_rubric.minimum_staleness import _score_prompt


def _binary_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "criterion_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "score": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["criterion_id", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criterion_scores"],
        "additionalProperties": False,
    }


def test_qwen_and_gpt_gold_share_paper_system_prompt_and_context() -> None:
    conversation = ({"role": "user", "content": "I have a fever. What should I do?"},)
    criterion = "The response recommends urgent care for severe warning signs."
    response = "Seek urgent care for trouble breathing or confusion."

    qwen_prompt = qwen_criterion_prompt(conversation, response, criterion)
    assert PAPER_JUDGE_SYSTEM_PROMPT in qwen_prompt
    assert "USER:\nI have a fever" in qwen_prompt
    assert qwen_prompt.endswith("Answer:")
    assert (
        _score_prompt(
            criterion,
            response,
            conversation=conversation,
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        == qwen_prompt
    )

    line, identity = _request(
        "prompt-1",
        "response-1",
        response,
        [{"criterion": criterion, "points": 1}],
        _binary_schema(),
        conversation=conversation,
        prompt_version=PAPER_JUDGE_PROMPT_VERSION,
    )
    messages = line["body"]["input"]
    assert messages[0] == {"role": "system", "content": PAPER_JUDGE_SYSTEM_PROMPT}
    assert "USER:\nI have a fever" in messages[1]["content"]
    assert "Do not use intermediate values" in messages[1]["content"]
    assert line["body"]["text"]["format"]["schema"] == _binary_schema()
    assert identity["grader_prompt_version"] == PAPER_JUDGE_PROMPT_VERSION
    assert len(identity["conversation_hash"]) == 64


def test_paper_gold_user_payload_contains_every_physician_criterion() -> None:
    criteria = [
        {"criterion": "Includes warning signs", "points": 2},
        {"criterion": "Does not claim certainty", "points": -1},
    ]
    line, _ = _request(
        "prompt-1",
        "response-1",
        "A cautious answer.",
        criteria,
        _binary_schema(),
        conversation=({"role": "user", "content": "Question"},),
        prompt_version=PAPER_JUDGE_PROMPT_VERSION,
    )
    content = line["body"]["input"][1]["content"]
    encoded = json.loads(content.split("Evaluation criteria:\n", 1)[1].split("\n\n", 1)[0])
    assert [item["criterion_id"] for item in encoded] == ["gold-000", "gold-001"]
