#!/usr/bin/env python3
"""Offline sanity check for an ACP checkpoint on a processed zarr dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def default_acp_root() -> Path:
    return Path(__file__).resolve().parents[2] / "adaptive_compliance_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an ACP checkpoint, run policy.predict_action() on samples from a "
            "processed ACP zarr dataset, and report action MSE."
        )
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Checkpoint .ckpt file, or a training run folder containing checkpoints/latest.ckpt.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Processed ACP zarr dataset folder, for example $PYRITE_DATASET_FOLDERS/pick_0706_v1.",
    )
    parser.add_argument(
        "--acp-root",
        type=Path,
        default=default_acp_root(),
        help="Path to adaptive_compliance_policy.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix == ".ckpt":
        return path
    return path / "checkpoints" / "latest.ckpt"


def choose_sample_indices(dataset_len: int, num_samples: int, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("dataset is empty")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    count = min(num_samples, dataset_len)
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_len, size=count, replace=False)
    return sorted(int(x) for x in indices)


def compute_action_metrics(pred: Any, target: Any) -> dict[str, float]:
    pred_np = to_numpy(pred).astype(np.float64)
    target_np = to_numpy(target).astype(np.float64)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"shape mismatch: pred {pred_np.shape}, target {target_np.shape}")
    if pred_np.shape[-1] not in (19, 38):
        raise ValueError(f"expected action dim 19 or 38, got {pred_np.shape[-1]}")

    diff = pred_np - target_np
    metrics = {"mse/all": float(np.mean(diff * diff))}
    per_arm_dim = 19
    num_arms = pred_np.shape[-1] // per_arm_dim
    for arm in range(num_arms):
        offset = arm * per_arm_dim
        prefix = "" if num_arms == 1 else f"arm{arm}/"
        metrics[f"mse/{prefix}reference_pose9"] = float(
            np.mean(diff[..., offset : offset + 9] ** 2)
        )
        metrics[f"mse/{prefix}virtual_target_pose9"] = float(
            np.mean(diff[..., offset + 9 : offset + 18] ** 2)
        )
        metrics[f"mse/{prefix}stiffness"] = float(
            np.mean(diff[..., offset + 18 : offset + 19] ** 2)
        )
    return metrics


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_acp_to_sys_path(acp_root: Path) -> None:
    acp_root = acp_root.expanduser().resolve()
    candidates = [
        acp_root,
        acp_root / "PyriteML",
        acp_root / "PyriteConfig",
        acp_root / "PyriteEnvSuites",
        acp_root / "PyriteUtility",
    ]
    for path in candidates:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def dict_to_device(data: Any, device: str) -> Any:
    import torch

    if isinstance(data, dict):
        return {key: dict_to_device(value, device) for key, value in data.items()}
    if torch.is_tensor(data):
        return data.to(device)
    return data


def stack_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    return default_collate(samples)


def load_checkpoint_cfg(ckpt_path: Path):
    import dill
    import torch

    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    return payload["cfg"]


def load_dataset(cfg: Any, dataset_path: Path):
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.task.dataset.dataset_path = str(dataset_path.expanduser().resolve())
    return hydra.utils.instantiate(cfg.task.dataset)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    add_acp_to_sys_path(args.acp_root)

    from PyriteUtility.pytorch_utils.model_io import load_policy

    ckpt_path = resolve_checkpoint_path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    cfg = load_checkpoint_cfg(ckpt_path)
    dataset = load_dataset(cfg, dataset_path)
    indices = choose_sample_indices(len(dataset), args.num_samples, args.seed)
    policy, _ = load_policy(str(ckpt_path), args.device)

    import torch

    all_metrics = []
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        samples = [dataset[idx] for idx in batch_indices]
        batch = stack_samples(samples)
        batch = dict_to_device(batch, args.device)
        with torch.no_grad():
            pred_action = policy.predict_action(batch["obs"])
        metrics = compute_action_metrics(pred_action["sparse"], batch["action"]["sparse"])
        metrics["batch_size"] = len(batch_indices)
        all_metrics.append(metrics)

    weighted = {}
    total = sum(item["batch_size"] for item in all_metrics)
    for key in all_metrics[0]:
        if key == "batch_size":
            continue
        weighted[key] = float(
            sum(item[key] * item["batch_size"] for item in all_metrics) / total
        )

    result = {
        "checkpoint": str(ckpt_path),
        "dataset": str(dataset_path),
        "num_dataset_samples": len(dataset),
        "num_eval_samples": total,
        "indices": indices,
        "metrics": weighted,
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
