from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_perm1_fastpath.py"
_SPEC = importlib.util.spec_from_file_location("run_perm1_fastpath", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
permutation_zero_rows = _MODULE.permutation_zero_rows


def test_permutation_zero_rows_filters_without_changing_selection_fields() -> None:
    rows = [
        {"mode": "static", "n": 1, "permutation": 0, "response_id": "a"},
        {"mode": "static", "n": 1, "permutation": 1, "response_id": "b"},
        {"mode": "dynamic_fixed_budgeted", "n": 1, "permutation": 0, "response_id": "c"},
    ]

    selected = permutation_zero_rows(rows)

    assert selected == [rows[0], rows[2]]
