"""Lossless common-state checkpoints and exact continuation fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def _scalar_fingerprint(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float64:{struct.pack('>d', value).hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, bytes):
        return f"bytes:{hashlib.sha256(value).hexdigest()}"
    if isinstance(value, np.generic):
        return f"numpy_scalar:{value.dtype}:{value.tobytes().hex()}"
    raise TypeError(f"unsupported fingerprint scalar: {type(value).__name__}")


def flatten_fingerprints(value: Any, path: str = "root") -> Dict[str, str]:
    """Produce exact, path-addressable fingerprints for diagnostic comparison."""
    result: Dict[str, str] = {}
    if isinstance(value, torch.Tensor):
        result[path] = (
            f"tensor:{value.dtype}:{list(value.shape)}:{tensor_sha256(value)}"
        )
    elif isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
        result[path] = f"ndarray:{value.dtype}:{list(value.shape)}:{digest}"
    elif isinstance(value, Mapping):
        result[path] = f"mapping:{len(value)}"
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            key_name = f"{type(key).__name__}:{repr(key)}"
            result.update(flatten_fingerprints(value[key], f"{path}/{key_name}"))
    elif isinstance(value, (list, tuple)):
        result[path] = f"{type(value).__name__}:{len(value)}"
        for index, item in enumerate(value):
            result.update(flatten_fingerprints(item, f"{path}/{index}"))
    elif isinstance(value, set):
        ordered = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        result[path] = f"set:{len(ordered)}"
        for index, item in enumerate(ordered):
            result.update(flatten_fingerprints(item, f"{path}/{index}"))
    else:
        result[path] = _scalar_fingerprint(value)
    return result


def fingerprint_sha256(value: Any) -> str:
    flat = flatten_fingerprints(value)
    encoded = json.dumps(flat, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_fingerprints(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> list[Dict[str, str]]:
    differences = []
    for path in sorted(set(expected) | set(actual)):
        left = expected.get(path, "<missing>")
        right = actual.get(path, "<missing>")
        if left != right:
            differences.append({"path": path, "expected": left, "actual": right})
    return differences


def capture_rng_state(generator: torch.Generator) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "data_loader_generator": generator.get_state().clone(),
    }


def restore_rng_state(state: Mapping[str, Any], generator: torch.Generator) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "data_loader_generator"}
    if set(state) != required:
        raise ValueError("invalid RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state["data_loader_generator"])


def _git_value(args: Iterable[str], default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def protocol_manifest(
    protocol: Mapping[str, Any], code_paths: Iterable[os.PathLike[str] | str]
) -> Dict[str, Any]:
    code_hashes = {
        str(Path(path)): sha256_file(path)
        for path in code_paths
        if Path(path).is_file()
    }
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "protocol": dict(protocol),
        "code": {
            "git_commit": _git_value(["rev-parse", "HEAD"], "unavailable"),
            "git_status": _git_value(["status", "--short"], "unavailable"),
            "files_sha256": code_hashes,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def save_checkpoint(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite immutable checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    metadata = {
        "checkpoint": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "starting_state_hash": fingerprint_sha256(payload),
        "next_round": int(payload["next_round"]),
        "parent_run_id": str(payload["parent_run_id"]),
        "manifest": payload["manifest"],
    }
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def load_checkpoint(path: os.PathLike[str] | str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing checkpoint manifest {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    checksum = sha256_file(path)
    if checksum != metadata.get("sha256"):
        raise RuntimeError(
            f"checkpoint checksum mismatch: expected {metadata.get('sha256')}, got {checksum}"
        )
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    return payload, metadata


def validate_resume_protocol(
    checkpoint_protocol: Mapping[str, Any], current_protocol: Mapping[str, Any]
) -> None:
    """Allow only projection -> projection/shrinkage treatment changes on resume."""
    checkpoint = dict(checkpoint_protocol)
    current = dict(current_protocol)
    checkpoint_treatment = checkpoint.pop("update_control", None)
    current_treatment = current.pop("update_control", None)
    if checkpoint_treatment != "projection":
        raise RuntimeError(
            "resume checkpoint must have update_control='projection', got "
            f"{checkpoint_treatment!r}"
        )
    if current_treatment not in {"projection", "shrinkage", "shrinkage_online"}:
        raise RuntimeError(f"invalid resumed update_control {current_treatment!r}")
    if checkpoint != current:
        missing = object()
        differing = {
            key: {
                "checkpoint": (
                    "<missing>" if checkpoint.get(key, missing) is missing else checkpoint[key]
                ),
                "current": (
                    "<missing>" if current.get(key, missing) is missing else current[key]
                ),
            }
            for key in sorted(set(checkpoint) | set(current))
            if checkpoint.get(key, missing) != current.get(key, missing)
        }
        raise RuntimeError(
            "resume protocol differs from checkpoint outside update_control: "
            f"{differing}"
        )


def requires_endpoint_equality(update_control: str, projection_lambda: float) -> bool:
    """Only the no-treatment projection branch is an exact parent continuation."""
    return update_control == "projection" and projection_lambda == 0.0


def validate_determinism_gate(
    gate_path: os.PathLike[str] | str,
    *,
    parent_run_id: str,
    executed_round: int,
    source_checkpoint_sha256: str,
) -> Dict[str, Any]:
    """Validate the one exact-continuation gate relevant to a resumed checkpoint."""
    gate_path = Path(gate_path)
    if not gate_path.is_file():
        raise RuntimeError(f"branch blocked: missing determinism gate {gate_path}")
    document = json.loads(gate_path.read_text(encoding="utf-8"))
    if not document.get("determinism_gate", {}).get("passed", False):
        raise RuntimeError(f"branch blocked: determinism gate did not pass {gate_path}")
    expected = {
        "parent_run_id": str(parent_run_id),
        "executed_round": int(executed_round),
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
    }
    actual = {key: document.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "branch blocked: determinism gate belongs to another checkpoint: "
            f"expected={expected}, found={actual}"
        )
    return document


def branch_control_metadata(
    *,
    update_control: str,
    branch_local_epochs: int | None,
    shrinkage_schedule: os.PathLike[str] | str | None,
    client_epoch_counts: Mapping[int, int],
    client_optimizer_step_counts: Mapping[int, int],
) -> Dict[str, Any]:
    """Build auditable treatment/compute metadata for a one-round branch."""
    schedule_path = str(Path(shrinkage_schedule)) if shrinkage_schedule else None
    return {
        "update_control": update_control,
        "branch_local_epochs": branch_local_epochs,
        "shrinkage_schedule_path": schedule_path,
        "shrinkage_schedule_sha256": (
            sha256_file(shrinkage_schedule) if shrinkage_schedule else None
        ),
        "per_client_epoch_counts": {
            str(key): int(value) for key, value in client_epoch_counts.items()
        },
        "per_client_optimizer_step_counts": {
            str(key): int(value)
            for key, value in client_optimizer_step_counts.items()
        },
    }


def endpoint_state(
    global_model,
    clients,
    subspace_instrumentation,
    summary_metrics,
    history_state=None,
    rng_state=None,
):
    state = {
        "global_model": global_model.state_dict(),
        "optimizers": [client.optimizer.state_dict() for client in clients],
        "replay_buffers": [client.replay.state_dict() for client in clients],
        "client_persistent_state": [
            client.persistent_state_dict() for client in clients
        ],
        "subspace": subspace_instrumentation.state_dict(),
        "summary_metrics": dict(summary_metrics),
    }
    if history_state is not None:
        state["history_state"] = history_state
    if rng_state is not None:
        state["rng_state"] = rng_state
    return state


def write_endpoint_reference(
    path: os.PathLike[str] | str,
    endpoint: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite endpoint reference {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = flatten_fingerprints(endpoint)
    document = {
        **dict(metadata),
        "endpoint_state_hash": fingerprint_sha256(endpoint),
        "endpoint_global_model_hash": model_state_sha256(endpoint["global_model"]),
        "fingerprints": fingerprints,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return document


def verify_endpoint_reference(
    reference_path: os.PathLike[str] | str,
    endpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    reference_path = Path(reference_path)
    expected = json.loads(reference_path.read_text(encoding="utf-8"))
    actual_fingerprints = flatten_fingerprints(endpoint)
    differences = compare_fingerprints(expected["fingerprints"], actual_fingerprints)
    return {
        "passed": not differences,
        "reference": str(reference_path),
        "expected_endpoint_state_hash": expected["endpoint_state_hash"],
        "actual_endpoint_state_hash": fingerprint_sha256(endpoint),
        "differences": differences,
    }
