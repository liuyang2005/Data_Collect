import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "postprocess" / "eval_acp_checkpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_acp_checkpoint", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_sample_indices_is_deterministic_and_sorted():
    module = load_module()

    first = module.choose_sample_indices(dataset_len=10, num_samples=4, seed=123)
    second = module.choose_sample_indices(dataset_len=10, num_samples=4, seed=123)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 4
    assert all(0 <= idx < 10 for idx in first)


def test_compute_action_metrics_splits_action19_components():
    module = load_module()
    gt = np.zeros((2, 3, 19), dtype=np.float32)
    pred = np.zeros_like(gt)
    pred[..., 0:9] = 1.0
    pred[..., 9:18] = 2.0
    pred[..., 18:19] = 3.0

    metrics = module.compute_action_metrics(pred, gt)

    assert metrics["mse/all"] == 2.8421052631578947
    assert metrics["mse/reference_pose9"] == 1.0
    assert metrics["mse/virtual_target_pose9"] == 4.0
    assert metrics["mse/stiffness"] == 9.0
