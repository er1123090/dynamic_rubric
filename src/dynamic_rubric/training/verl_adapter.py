from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import write_json_atomic


class DependencyGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerlCapabilities:
    checkout: str
    revision: str
    custom_reward_hook: bool
    raw_validation_export: bool
    focal_checkpoint_retention: bool
    probe_patch_required: bool
    stage5_patch_applied: bool
    patch_sha256: str | None = None
    online_patch_applied: bool = False
    online_patch_sha256: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.custom_reward_hook
            and self.focal_checkpoint_retention
            and self.stage5_patch_applied
            and bool(self.patch_sha256)
            and bool(
                self.raw_validation_export or (self.probe_patch_required and self.patch_sha256)
            )
        )


    @property
    
    def online_ready(self) -> bool:
        return self.ready and self.online_patch_applied and bool(self.online_patch_sha256)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkout_head(checkout: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DependencyGateError(
            f"cannot resolve pinned veRL checkout HEAD: {checkout}"
        ) from exc


def _patch_bundle_is_applied(checkout: Path, patches: tuple[Path, ...]) -> bool:
    """Build the expected checkout from HEAD plus ordered patches and compare targets."""

    targets: set[str] = set()
    for patch in patches:
        for line in patch.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ b/"):
                targets.add(line[6:])
    if not targets:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="verl-patch-gate-") as directory:
            expected_root = Path(directory)
            for relative in sorted(targets):
                destination = expected_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = subprocess.run(
                    ["git", "show", f"HEAD:{relative}"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                ).stdout
                destination.write_bytes(content)
            for patch in patches:
                subprocess.run(
                    ["git", "apply", "--unsafe-paths", str(patch.resolve())],
                    cwd=expected_root,
                    check=True,
                    capture_output=True,
                )
            return all(
                (expected_root / relative).read_bytes()
                == (checkout / relative).read_bytes()
                for relative in targets
            )
    except (OSError, subprocess.CalledProcessError):
        return False


def dependency_gate(lock: Mapping[str, Any], project_root: Path) -> VerlCapabilities:
    verl = lock.get("verl", {})
    checkout = Path(str(verl.get("checkout", "")))
    revision = str(verl.get("commit", ""))
    if not revision or revision.upper().startswith("UNPINNED"):
        raise DependencyGateError("veRL commit is not pinned")
    if not checkout.is_absolute():
        checkout = (project_root / checkout).resolve()
    if not checkout.is_dir():
        raise DependencyGateError(f"pinned veRL checkout is absent: {checkout}")
    checkout_head = _checkout_head(checkout)
    if checkout_head != revision:
        raise DependencyGateError(
            "pinned veRL checkout HEAD does not match lock: "
            f"expected {revision}, found {checkout_head}"
        )
    reward_file = checkout / "verl" / "trainer" / "ppo" / "reward.py"
    custom_reward = reward_file.is_file() and "custom_reward" in reward_file.read_text(
        encoding="utf-8", errors="ignore"
    )
    validation_candidates = tuple(checkout.rglob("*validation*.py"))
    raw_export = any(
        "response" in path.read_text(encoding="utf-8", errors="ignore")
        for path in validation_candidates
    )
    patch = project_root / "patches" / "verl_stage5_determinism.patch"
    patch_hash = sha256_file(patch) if patch.is_file() else None
    expected_patch_hash = str(verl.get("stage5_patch_sha256", ""))
    online_patch = project_root / "patches" / "verl_online_rubrics.patch"
    online_patch_hash = sha256_file(online_patch) if online_patch.is_file() else None
    expected_online_patch_hash = str(verl.get("online_patch_sha256", ""))
    patch_hashes_match = bool(patch_hash and patch_hash == expected_patch_hash)
    online_hashes_match = bool(
        online_patch_hash
        and expected_online_patch_hash
        and online_patch_hash == expected_online_patch_hash
    )
    bundle = (patch, online_patch) if online_hashes_match else (patch,)
    bundle_applied = patch_hashes_match and _patch_bundle_is_applied(checkout, bundle)
    patch_applied = bundle_applied
    online_patch_applied = bundle_applied and online_hashes_match
    # Checkpoint retention is a config-level capability in supported veRL pins;
    # the lock records the smoke result rather than inferring it from filenames.
    retention = bool(verl.get("focal_checkpoint_retention_verified", False))
    capabilities = VerlCapabilities(
        checkout=str(checkout),
        revision=revision,
        custom_reward_hook=custom_reward,
        raw_validation_export=raw_export,
        focal_checkpoint_retention=retention,
        probe_patch_required=not raw_export,
        stage5_patch_applied=patch_applied,
        patch_sha256=patch_hash,
        online_patch_applied=online_patch_applied,
        online_patch_sha256=online_patch_hash,
    )
    if not capabilities.ready:
        raise DependencyGateError(f"veRL capability gate failed: {asdict(capabilities)}")
    return capabilities


def write_launch_spec(
    output: Path,
    config_hash: str,
    run_id: str,
    reward_path: Path,
    capabilities: VerlCapabilities,
    training: Mapping[str, Any],
) -> None:
    value = {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": config_hash,
        "reward_source": "static_r0_only",
        "static_rubric_path": str(reward_path),
        "after_optimizer_update_semantics": True,
        "capabilities": asdict(capabilities),
        "training": dict(training),
    }
    write_json_atomic(output, value, immutable=True)
