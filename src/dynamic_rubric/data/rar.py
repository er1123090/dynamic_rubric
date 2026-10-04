"""Canonical RaR ingestion for the discriminability-horizon experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from ..hashing import canonical_json_bytes, sha256_file
from ..horizon.contracts import criterion_content_hash
from .splits import SplitSpec, assign_splits, validate_disjoint, write_splits


class RaRDataError(ValueError):
    pass


WEIGHT_UNITS = {"essential": 10, "important": 7, "optional": 3, "pitfall": 9}
_POSITIVE_PITFALL_RE = re.compile(
    r"^(?:(?:the (?:response|answer)\s+)?(?:should|must)\s+(?:avoid|not)\b|"
    r"(?:the (?:response|answer)\s+)?avoid(?:s|ing)?\b|does not\b|doesn't\b|"
    r"never\b|refrain(?:s|ing)?\b|omit(?:s|ting)?\b|prevent(?:s|ing)?\b|without\b)",
    re.IGNORECASE,
)
_IMPORTANCE_PREFIX_RE = re.compile(
    r"^\s*(essential|important|optional|pitfall)\s+criteria?\s*:\s*",
    re.IGNORECASE,
)


def _records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as error:
            raise RaRDataError(
                "reading official RaR parquet requires the veRL runtime with pyarrow"
            ) from error
        rows = parquet.read_table(path).to_pylist()
    elif path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            value = value.get("records", value.get("data"))
        rows = value
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise RaRDataError("RaR source must be a JSON array/records mapping or JSONL objects")
    return list(rows)


def _prompt_payload(row: Mapping[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        normalized: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, Mapping) or item.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                raise RaRDataError("messages must contain valid role/content mappings")
            content = str(item.get("content", "")).strip()
            if not content:
                raise RaRDataError("message content must be non-empty")
            normalized.append({"role": str(item["role"]), "content": content})
        if not normalized:
            raise RaRDataError("messages must be non-empty")
        return normalized, json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    prompt = str(row.get("prompt", row.get("question", ""))).strip()
    if not prompt:
        raise RaRDataError("row must contain non-empty messages, prompt, or question")
    return [{"role": "user", "content": prompt}], prompt


def _importance(item: Mapping[str, Any]) -> str:
    description = str(item.get("description", item.get("criterion", item.get("text", ""))))
    prefix = _IMPORTANCE_PREFIX_RE.match(description)
    if prefix is not None:
        return prefix.group(1).lower()
    kind = str(item.get("criterion_type", item.get("type", "quality"))).lower()
    if kind == "pitfall" or bool(item.get("pitfall", False)):
        return "pitfall"
    value = str(item.get("importance_class", item.get("importance", ""))).lower()
    if value in {"essential", "important", "optional"}:
        return value
    raw = item.get("weight", item.get("points"))
    numeric = float(raw) if raw is not None else None
    aliases = {1.0: "essential", 0.7: "important", 0.3: "optional", 0.9: "pitfall"}
    if numeric in aliases:
        return aliases[numeric]
    unit_aliases = {10.0: "essential", 7.0: "important", 3.0: "optional", 9.0: "pitfall"}
    if numeric in unit_aliases:
        return unit_aliases[numeric]
    if numeric is not None:
        if numeric < 0:
            return "pitfall"
        if numeric >= 5:
            return "essential"
        if numeric >= 3:
            return "important"
        if numeric >= 1:
            return "optional"
    raise RaRDataError("criterion needs an allowed importance class or exact RaR weight")


def _rubric_items(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("r0", row.get("initial_rubric", row.get("rubric", row.get("criteria"))))
    if isinstance(value, Mapping):
        value = value.get("criteria")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RaRDataError("row must contain a non-empty R0 rubric/criteria list")
    if not all(isinstance(item, Mapping) for item in value):
        raise RaRDataError("rubric criteria must be mappings")
    return list(value)


def normalize_rar_row(row: Mapping[str, Any], *, domain: str) -> dict[str, Any]:
    if domain not in {"medicine", "science"}:
        raise RaRDataError("domain must be medicine or science")
    messages, prompt_fingerprint = _prompt_payload(row)
    prompt_hash = hashlib.sha256(canonical_json_bytes(messages)).hexdigest()
    source_id = str(row.get("prompt_id", row.get("id", ""))).strip()
    if not source_id:
        digest = hashlib.sha256(
            canonical_json_bytes([domain, prompt_fingerprint])
        ).hexdigest()[:24]
        source_id = digest
    prompt_id = f"rar_{domain}_{source_id}"
    criteria: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(_rubric_items(row)):
        text = str(item.get("criterion", item.get("description", item.get("text", "")))).strip()
        if not text:
            raise RaRDataError("criterion text must be non-empty")
        text = _IMPORTANCE_PREFIX_RE.sub("", text).strip()
        if not text:
            raise RaRDataError("criterion text must remain non-empty after removing its label")
        importance = _importance(item)
        positive_form = item.get("positive_form")
        raw_weight = item.get("weight", item.get("points"))
        source_weight = float(raw_weight) if raw_weight is not None else None
        if importance == "pitfall":
            if positive_form is False:
                raise RaRDataError("pitfall criteria must be expressed as positive avoidance")
            if positive_form is not True and not _POSITIVE_PITFALL_RE.search(text):
                if source_weight is None or source_weight >= 0:
                    raise RaRDataError("pitfall criteria must be expressed as positive avoidance")
                text = f"Avoids this pitfall: {text}"
        canonical = " ".join(text.casefold().split())
        if canonical in seen:
            raise RaRDataError("R0 contains duplicate criterion text")
        seen.add(canonical)
        criterion_hash = criterion_content_hash(text)
        criteria.append(
            {
                "criterion_id": f"{prompt_id}:r0:{index}",
                "criterion": text,
                "importance_class": importance,
                "criterion_type": "pitfall" if importance == "pitfall" else "quality",
                "weight_units": WEIGHT_UNITS[importance],
                "canonical_criterion_hash": criterion_hash,
                "positive_form": True,
            }
        )
    return {
        "schema_version": 1,
        "prompt_id": prompt_id,
        "source_prompt_id": source_id,
        "domain": domain,
        "messages": messages,
        "prompt_hash": prompt_hash,
        "r0": {"criteria": criteria},
        "strata": (
            dict(row["strata"])
            if isinstance(row.get("strata"), Mapping)
            else (
                {"question_source": str(row["question_source"])}
                if str(row.get("question_source", "")).strip()
                else {}
            )
        ),
        "reference_answer": str(row.get("reference_answer", "")).strip(),
        "source": "rar",
    }


def normalize_rar_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    deduplicate_prompt_hashes: bool = False,
) -> list[dict[str, Any]]:
    normalized = [normalize_rar_row(row, domain=domain) for row in rows]
    if deduplicate_prompt_hashes:
        selected: dict[str, dict[str, Any]] = {}
        for row in normalized:
            prompt_hash = str(row["prompt_hash"])
            prior = selected.get(prompt_hash)
            if prior is None:
                selected[prompt_hash] = row
                continue
            row_rank = (-len(row["r0"]["criteria"]), canonical_json_bytes(row))
            prior_rank = (-len(prior["r0"]["criteria"]), canonical_json_bytes(prior))
            if row_rank < prior_rank:
                selected[prompt_hash] = row
        normalized = [selected[key] for key in sorted(selected)]
    prompt_ids = [row["prompt_id"] for row in normalized]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise RaRDataError("normalized RaR prompt IDs must be unique")
    prompt_hashes = [row["prompt_hash"] for row in normalized]
    if not deduplicate_prompt_hashes and len(prompt_hashes) != len(set(prompt_hashes)):
        raise RaRDataError("normalized RaR prompt hashes must be unique")
    return normalized


def _strata_balance(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    keys = sorted(
        {
            str(key)
            for rows in splits.values()
            for row in rows
            for key in row.get("strata", {})
        }
    )
    return {
        "keys": keys,
        "counts": {
            split: {
                key: dict(
                    sorted(
                        Counter(str(row.get("strata", {}).get(key, "__missing__")) for row in rows).items()
                    )
                )
                for key in keys
            }
            for split, rows in splits.items()
        },
    }


def prepare_rar_source(
    source: Path,
    output_dir: Path,
    *,
    domain: str,
    seed: int,
    train_count: int = 1500,
    development_count: int = 150,
    final_count: int = 300,
) -> dict[str, Any]:
    source_rows = _records(source)
    rows = normalize_rar_rows(
        source_rows,
        domain=domain,
        deduplicate_prompt_hashes=True,
    )
    specs = (
        SplitSpec("train", train_count),
        SplitSpec("development", development_count),
        SplitSpec("final", final_count),
    )
    splits = assign_splits(rows, specs, seed)
    validate_disjoint(splits)
    manifest = write_splits(splits, output_dir, seed)
    normalized_path = output_dir / "normalized.jsonl"
    write_jsonl_atomic(normalized_path, rows)
    manifest.update(
        {
            "schema_version": 2,
            "domain": domain,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_record_count": len(source_rows),
            "normalized_record_count": len(rows),
            "deduplicated_record_count": len(source_rows) - len(rows),
            "normalized_sha256": sha256_file(normalized_path),
            "stratification": _strata_balance(splits),
        }
    )
    write_json_atomic(output_dir / "rar_manifest.json", manifest)
    return manifest
