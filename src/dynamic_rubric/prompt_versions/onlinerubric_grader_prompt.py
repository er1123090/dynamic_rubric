"""Figure-10-style binary rubric grader prompt for OnlineRubrics training."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


ONLINERUBRIC_GRADER_SYSTEM_PROMPT = """You are an expert evaluator. You will be given a prompt, a response to the prompt, and a numbered list of rubrics. Evaluate the response against EACH rubric independently and determine whether the response satisfies it.

For every numbered rubric, output `PRESENT` if the response satisfies the rubric and `NOT_PRESENT` if it does not.

Pay careful attention to rubrics written as negative requirements. For example, if a rubric says that the response should not contain a behavior or should avoid a behavior, output `PRESENT` when the response successfully avoids that behavior and `NOT_PRESENT` when the prohibited behavior occurs.

If a rubric contains multiple sentences or multiple required conditions, the response must satisfy all of those conditions for the rubric to be `PRESENT`. However, phrases such as "such as", "for example", or "including" introduce illustrative examples rather than an exhaustive checklist; the response does not need to contain every listed example to satisfy the general requirement.

Return a JSON object whose keys are exactly the supplied rubric numbers and whose values are exactly `PRESENT` or `NOT_PRESENT`. Do not omit a rubric, add a rubric, return probabilities, or include any text outside the JSON object."""


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def build_onlinerubric_grader_messages(
    *,
    prompt: Sequence[Mapping[str, str]],
    response: str,
    criteria: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, str], ...]:
    """Build a single-response, full-rubric binary grading request."""

    if not prompt:
        raise ValueError("prompt must contain at least one message")
    if not response.strip():
        raise ValueError("response must be non-empty")
    if not criteria:
        raise ValueError("criteria must not be empty")
    criterion_ids = [str(item.get("criterion_id", "")) for item in criteria]
    if any(not item.strip() for item in criterion_ids):
        raise ValueError("every criterion must have a non-empty criterion_id")
    if len(criterion_ids) != len(set(criterion_ids)):
        raise ValueError("criterion IDs must be unique")
    numbered = {
        str(index): str(item.get("criterion", item.get("text", "")))
        for index, item in enumerate(criteria, 1)
    }
    if any(not text.strip() for text in numbered.values()):
        raise ValueError("every criterion must contain text")
    content = "\n\n".join(
        (
            "Prompt:\n" + _json_block(list(prompt)),
            "Response:\n" + response,
            "Numbered Rubrics:\n" + _json_block(numbered),
        )
    )
    return (
        {"role": "system", "content": ONLINERUBRIC_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    )


def onlinerubric_grader_schema(criterion_count: int) -> Mapping[str, Any]:
    """Build the Figure-10 direct-object schema for exactly ``1..N`` rubric keys."""

    if criterion_count < 1:
        raise ValueError("criterion_count must be positive")
    keys = [str(index) for index in range(1, criterion_count + 1)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": keys,
        "properties": {
            key: {"type": "string", "enum": ["PRESENT", "NOT_PRESENT"]} for key in keys
        },
    }
