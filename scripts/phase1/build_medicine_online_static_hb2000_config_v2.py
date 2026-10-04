#!/usr/bin/env python3
"""Run the HB2000 config builder with paths relative to the config directory."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SOURCE = Path(__file__).with_name("build_medicine_online_static_hb2000_config.py")
SPEC = importlib.util.spec_from_file_location("hb2000_builder", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODULE.relative_to_config = lambda path: os.path.relpath(path, MODULE.CONFIG_DIR)


if __name__ == "__main__":
    MODULE.main()
