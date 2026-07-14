#!/usr/bin/env python3
"""Shared helpers for Data_Collect post-processing tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MAIN_CAMERA_NAME = "cam_327322062498"
WRIST_CAMERA_NAME = "cam_260322274925_wrist"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REQUIRED_SESSION_FILES = (
    "tcps.npy",
    "tcps_timestamps_host_s.npy",
    "angles.npy",
    "angles_timestamps_host_s.npy",
    "ext_wrench_in_tcp.npy",
    "ext_wrench_in_tcp_timestamps_host_s.npy",
)


def is_session_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "tcps.npy").is_file()


def discover_sessions(source: Path | str) -> list[Path]:
    source = Path(source).expanduser().resolve()
    if is_session_dir(source):
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    sessions = sorted(p for p in source.iterdir() if is_session_dir(p))
    if not sessions:
        raise RuntimeError(f"No Data_Collect record_* sessions found under {source}")
    return sessions


def camera_dirs(session: Path) -> list[Path]:
    return sorted(
        p
        for p in Path(session).iterdir()
        if p.is_dir() and p.name.startswith("cam_") and (p / "color").is_dir()
    )


def select_camera_dir(session: Path, camera_name: str = MAIN_CAMERA_NAME) -> Path:
    session = Path(session)
    if camera_name:
        requested = camera_name if camera_name.startswith("cam_") else f"cam_{camera_name}"
        cam = session / requested
        if cam.is_dir():
            return cam
        if requested == MAIN_CAMERA_NAME:
            non_wrist = [p for p in camera_dirs(session) if not p.name.endswith("_wrist")]
            if len(non_wrist) == 1:
                return non_wrist[0]
        raise FileNotFoundError(f"Camera directory not found in {session}: {requested}")

    cams = camera_dirs(session)
    if len(cams) == 1:
        return cams[0]
    non_wrist = [p for p in cams if not p.name.endswith("_wrist")]
    if len(non_wrist) == 1:
        return non_wrist[0]
    names = ", ".join(p.name for p in cams) or "none"
    raise RuntimeError(f"Cannot choose camera automatically in {session}; candidates: {names}")


def sorted_numeric_files(path: Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> list[Path]:
    path = Path(path)
    suffixes = {s.lower() for s in suffixes}
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes]

    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.name)

    return sorted(files, key=key)


def load_npy(path: Path, *, ndim: int | None = None) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{path} must be {ndim}D, got shape {arr.shape}")
    return arr


def load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_1d(values: Sequence[float] | np.ndarray, name: str = "timestamps") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def ensure_2d(values: np.ndarray, name: str = "array") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def estimate_rate_hz(timestamps_s: Sequence[float] | np.ndarray) -> float | None:
    ts = ensure_1d(timestamps_s)
    if len(ts) < 2:
        return None
    duration = float(ts[-1] - ts[0])
    if duration <= 0:
        return None
    return float((len(ts) - 1) / duration)


def nearest_indices(sample_times: Sequence[float] | np.ndarray, query_times: Sequence[float] | np.ndarray) -> np.ndarray:
    sample = ensure_1d(sample_times, "sample_times")
    query = ensure_1d(query_times, "query_times")
    if len(sample) == 0:
        raise ValueError("sample_times is empty")
    if len(sample) == 1:
        return np.zeros(len(query), dtype=np.int64)
    if np.any(np.diff(sample) < 0):
        raise ValueError("sample_times must be sorted increasingly")
    right = np.searchsorted(sample, query, side="left")
    right = np.clip(right, 1, len(sample) - 1)
    left = right - 1
    use_right = np.abs(sample[right] - query) < np.abs(query - sample[left])
    return np.where(use_right, right, left).astype(np.int64)


def alignment_stats_ms(reference_times_s: Sequence[float] | np.ndarray, query_times_s: Sequence[float] | np.ndarray) -> dict:
    ref = ensure_1d(reference_times_s, "reference_times_s")
    query = ensure_1d(query_times_s, "query_times_s")
    idx = nearest_indices(ref, query)
    errors_ms = (query - ref[idx]) * 1000.0
    abs_ms = np.abs(errors_ms)
    if len(abs_ms) == 0:
        return {"count": 0}
    return {
        "count": int(len(abs_ms)),
        "mean_abs_ms": float(np.mean(abs_ms)),
        "median_abs_ms": float(np.median(abs_ms)),
        "p95_abs_ms": float(np.percentile(abs_ms, 95)),
        "max_abs_ms": float(np.max(abs_ms)),
        "signed_mean_ms": float(np.mean(errors_ms)),
    }


def percentile_dict(values: Sequence[float] | np.ndarray, percentiles=(50, 90, 95, 97.5, 99, 100)) -> dict:
    arr = ensure_1d(values, "values")
    if len(arr) == 0:
        return {}
    return {str(p): float(np.percentile(arr, p)) for p in percentiles}


def session_arrays(session: Path) -> dict:
    session = Path(session)
    return {
        "tcps": load_npy(session / "tcps.npy", ndim=2),
        "tcp_ts": load_npy(session / "tcps_timestamps_host_s.npy", ndim=1),
        "angles": load_npy(session / "angles.npy", ndim=2),
        "angle_ts": load_npy(session / "angles_timestamps_host_s.npy", ndim=1),
        "wrench": load_npy(session / "ext_wrench_in_tcp.npy", ndim=2),
        "wrench_ts": load_npy(session / "ext_wrench_in_tcp_timestamps_host_s.npy", ndim=1),
        "metadata": load_json(session / "metadata.json"),
    }


def camera_arrays(session: Path, camera_name: str = MAIN_CAMERA_NAME) -> dict:
    cam = select_camera_dir(session, camera_name)
    color_files = sorted_numeric_files(cam / "color")
    depth_files = sorted_numeric_files(cam / "depth") if (cam / "depth").is_dir() else []
    ts = load_npy(cam / "timestamps_host_s.npy", ndim=1)
    return {
        "camera_dir": cam,
        "camera_name": cam.name,
        "color_files": color_files,
        "depth_files": depth_files,
        "camera_ts": ts,
    }
