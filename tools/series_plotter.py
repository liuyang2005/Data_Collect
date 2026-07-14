#!/usr/bin/env python3
"""Small time-series plotting helper adapted from Forcemimic2Pipeline Visualize/SeriesPlot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _ensure_2d_cols(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _ensure_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _strictly_increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).copy()
    if len(t) <= 1:
        return t
    span = float(np.nanmax(t) - np.nanmin(t)) or 1.0
    min_dt = span / max(len(t) * 2, 1)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + min_dt
    return t


def load_and_validate_series(spec: dict, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "data" in spec and spec["data"] is not None:
        data = spec["data"]
    else:
        data_path = Path(spec["data_path"])
        if not data_path.is_file():
            raise FileNotFoundError(f"[{name}] missing data: {data_path}")
        data = np.load(data_path)
    data = _ensure_2d_cols(data)

    if "timestamps" in spec and spec["timestamps"] is not None:
        ts = spec["timestamps"]
    else:
        ts_path = Path(spec["ts_path"])
        if not ts_path.is_file():
            raise FileNotFoundError(f"[{name}] missing timestamps: {ts_path}")
        ts = np.load(ts_path)
    ts = _ensure_1d(ts)

    if len(ts) != len(data):
        raise ValueError(f"[{name}] timestamp length {len(ts)} != data rows {len(data)}")

    labels = list(spec.get("labels") or [f"col_{i}" for i in range(data.shape[1])])
    if len(labels) != data.shape[1]:
        raise ValueError(f"[{name}] label count {len(labels)} != columns {data.shape[1]}")
    return data, ts, labels


def plot_timeseries(
    series_list: Sequence[dict],
    title: Optional[str] = None,
    time_unit: str = "s",
    save_path: Optional[str | Path] = None,
    show: bool = False,
    dpi: int = 140,
    figsize: Optional[tuple[float, float]] = None,
):
    if not series_list:
        raise ValueError("series_list must not be empty")
    loaded = []
    for i, spec in enumerate(series_list):
        name = spec.get("name", f"series_{i}")
        loaded.append((*load_and_validate_series(spec, name), name))

    rows = len(loaded)
    if figsize is None:
        figsize = (15.0, max(2.1 * rows, 4.0))
    fig, axes = plt.subplots(rows, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, (data, ts, labels, name) in zip(axes, loaded):
        t_rel_s = ts - ts[0]
        x = t_rel_s * 1000.0 if time_unit == "ms" else t_rel_s
        x = _strictly_increasing(x)
        for col, label in enumerate(labels):
            ax.plot(x, data[:, col], label=label, linewidth=1.1, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel(f"time ({time_unit})")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=7, ncol=min(4, len(labels)))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98] if title else None)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
