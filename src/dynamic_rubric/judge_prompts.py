"""Shared paper-style rubric judge prompts for Qwen proxy and GPT-5 gold audit."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


PAPER_JUDGE_PROMPT_VERSION = "paper-rubric-judge-v1"
PAPER_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of assistant responses.

You will be given:
1. the user conversation,
2. a generated assistant response, and
3. one or more evaluation criteria.

For each criterion, determine whether the property described by the criterion is present in
the generated response, given the context of the user conversation.

Evaluation rules:
- Evaluate each criterion independently.
- Assign YES only when the generated response clearly satisfies or exhibits the criterion.
- Assign NO when the criterion is not satisfied, is contradicted, or there is insufficient
  evidence in the response.
- If a criterion contains multiple required parts, assign YES only when all required parts
  are satisfied.
- Examples introduced by phrases such as "for example" or "such as" are illustrative and
  are not necessarily exhaustive.
- Apply the criterion literally, regardless of whether it describes desirable or undesirable
  behavior.
- Use the user conversation to interpret the request, relevant context, and
  instruction-following requirements.
- Use your expert knowledge when the criterion requires evaluating factual correctness.
- Judge only the generated response as written. Do not give credit for information that is
  omitted or merely implied.
- Do not compare the response with other candidate responses.
- Follow the exact output format requested by the user message.
- Do not provide explanations, reasoning, or additional text."""


def render_conversation(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render the immutable HealthBench conversation with explicit role boundaries."""

    rendered: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError(f"unsupported conversation role at index {index}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"empty conversation content at index {index}")
        rendered.append(f"{str(role).upper()}:\n{content}")
    if not rendered:
        raise ValueError("conversation must not be empty")
    return "\n\n".join(rendered)


def qwen_criterion_prompt(
    messages: Sequence[Mapping[str, Any]], response_text: str, criterion_text: str
) -> str:
    """Render one binary criterion decision for Qwen target likelihood scoring."""

    if not response_text.strip() or not criterion_text.strip():
        raise ValueError("response and criterion must not be empty")
    return (
        "System instructions:\n"
        + PAPER_JUDGE_SYSTEM_PROMPT
        + "\n\nUser conversation:\n"
        + render_conversation(messages)
        + "\n\nGenerated assistant response:\n"
        + response_text
        + "\n\nEvaluation criterion:\n"
        + criterion_text
        + "\n\nReturn exactly one label: YES or NO.\n\nAnswer:"
    )


def gold_judge_user_prompt(
    messages: Sequence[Mapping[str, Any]],
    response_text: str,
    criteria: Sequence[Mapping[str, str]],
) -> str:
    """Render the GPT-5 user message for batched binary physician criteria."""

    if not response_text.strip() or not criteria:
        raise ValueError("response and criteria must not be empty")
    return (
        "User conversation:\n"
        + render_conversation(messages)
        + "\n\nGenerated assistant response:\n"
        + response_text
        + "\n\nEvaluation criteria:\n"
        + json.dumps(list(criteria), ensure_ascii=False, sort_keys=True)
        + "\n\nReturn every supplied criterion_id exactly once. Set score to 1 for YES and "
        "0 for NO. Do not use intermediate values."
    )
