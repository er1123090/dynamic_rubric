"""Blinded, paper-compatible horizon extraction request construction."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from ..prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from ..rubrics.extractor import make_blind_pairing


EXTRACTION_SCHEMA_VERSION = "horizon_extraction_v1"
DEDUP_SCHEMA_VERSION = "horizon_dedup_v1"
EXTRACTION_SCHEMA_SUFFIX = """

Horizon Structured Output extension:
- candidate_id: a unique stable ID for this request.
- quote: an exact substring of Response A or Response B.
- criterion: a positive, atomic, self-contained binary-checkable statement.
- weight: a positive integer following the paper prompt.
- importance_class: exactly essential, important, or optional.
- criterion_type: quality or pitfall. A pitfall must be written as positive avoidance.
Return only fields allowed by the supplied JSON schema.
""".strip()
DEDUP_SCHEMA_SUFFIX = """

Horizon Structured Output extension:
- Do not invent, merge into new wording, or rewrite a criterion.
- criterion must exactly copy one source candidate criterion.
- source_candidate_ids must list every candidate represented by that retained criterion.
- Do not output weight; deterministic code resolves class and weight from source candidates.
Return only fields allowed by the supplied JSON schema.
""".strip()


def _append_system_suffix(
    messages: Sequence[Mapping[str, str]], suffix: str
) -> tuple[Mapping[str, str], ...]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("paper prompt must begin with a system message")
    first = {**messages[0], "content": str(messages[0]["content"]) + "\n\n" + suffix}
    return (first, *messages[1:])


def prepare_extraction_requests(
    *,
    prompt_id: str,
    checkpoint_id: str,
    prompt: Sequence[Mapping[str, str]],
    existing_r0: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    pairing_seed: int | str,
) -> tuple[dict[str, Any], ...]:
    """Create exactly eight source-blind Figure-8 requests for one audit cell."""

    if len(current_rows) != 8 or len(control_rows) != 8:
        raise ValueError("horizon extraction requires exactly eight current/control responses")
    if any(str(row.get("prompt_id")) != prompt_id for row in (*current_rows, *control_rows)):
        raise ValueError("all extraction responses must belong to the requested prompt")
    pairing = make_blind_pairing(
        [str(row["response_text"]) for row in current_rows],
        [str(row["response_text"]) for row in control_rows],
        seed=pairing_seed,
        prompt_id=prompt_id,
        step=int(checkpoint_id.removeprefix("step")),
    )
    requests = []
    for pair in pairing.generator_payload():
        messages = _append_system_suffix(
            build_onlinerubric_extractor_messages(
                prompt=prompt,
                existing_rubric=existing_r0,
                response_a=pair.response_a,
                response_b=pair.response_b,
            ),
            EXTRACTION_SCHEMA_SUFFIX,
        )
        requests.append(
            {
                "schema_version": EXTRACTION_SCHEMA_VERSION,
                "request_id": "hex_"
                + sha256(f"{prompt_id}\x1f{checkpoint_id}\x1f{pair.pair_id}".encode()).hexdigest(),
                "prompt_id": prompt_id,
                "checkpoint_id": checkpoint_id,
                "source_pair_id": pair.pair_id,
                "response_a": pair.response_a,
                "response_b": pair.response_b,
                "messages": list(messages),
            }
        )
    if len({row["source_pair_id"] for row in requests}) != 8:
        raise AssertionError("extraction pair identities are not unique")
    forbidden = ("current_label", "control_label", "current_index", "control_index")
    if any(field in row for row in requests for field in forbidden):
        raise AssertionError("generator payload leaked response source identity")
    return tuple(requests)


def prepare_dedup_request(
    *,
    prompt_id: str,
    checkpoint_id: str,
    prompt: Sequence[Mapping[str, str]],
    existing_r0: Sequence[Mapping[str, Any]],
    candidate_criteria: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [str(item["candidate_id"]) for item in candidate_criteria]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    return {
        "schema_version": DEDUP_SCHEMA_VERSION,
        "prompt_id": prompt_id,
        "checkpoint_id": checkpoint_id,
        "allowed_source_candidate_ids": sorted(candidate_ids),
        "messages": list(
            _append_system_suffix(
                build_onlinerubric_dedup_messages(
                    prompt=prompt,
                    existing_rubric=existing_r0,
                    candidate_criteria=candidate_criteria,
                ),
                DEDUP_SCHEMA_SUFFIX,
            )
        ),
    }
