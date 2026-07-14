import importlib
import json
import sys
from pathlib import Path

import numpy as np


POSTPROCESS_DIR = Path(__file__).resolve().parents[1] / "postprocess"
if str(POSTPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_DIR))


def import_convert_acp_raw():
    sys.modules.pop("convert_acp_raw", None)
    return importlib.import_module("convert_acp_raw")


def write_fake_images(camera_dir: Path, count: int) -> None:
    color_dir = camera_dir / "color"
    depth_dir = camera_dir / "depth"
    color_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    for idx in range(count):
        (color_dir / f"{idx:016d}.png").write_bytes(f"color-{idx}".encode("ascii"))
        (depth_dir / f"{idx:016d}.png").write_bytes(f"depth-{idx}".encode("ascii"))


def write_common_arrays(session_dir: Path, legacy: bool = False) -> None:
    robot_dir = session_dir if legacy else session_dir / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    pose_name = "tcps.npy" if legacy else "tcp_pose.npy"
    timestamp_name = (
        "tcps_timestamps_host_s.npy" if legacy else "timestamps_host_s.npy"
    )
    np.save(
        robot_dir / pose_name,
        np.array(
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.08],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.7, 0.04],
            ],
            dtype=np.float64,
        ),
    )
    np.save(
        robot_dir / timestamp_name,
        np.array([100.00, 100.05], dtype=np.float64),
    )
    if not legacy:
        np.save(robot_dir / "tcp_vel.npy", np.zeros((2, 6), dtype=np.float64))
        np.save(robot_dir / "q.npy", np.zeros((2, 8), dtype=np.float64))
    np.save(
        session_dir / "ext_wrench_in_tcp.npy",
        np.array(
            [
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            ],
            dtype=np.float64,
        ),
    )
    np.save(
        session_dir / "ext_wrench_in_tcp_timestamps_host_s.npy",
        np.array([99.99, 100.02, 100.07], dtype=np.float64),
    )


def test_convert_session_writes_acp_raw_episode_with_relative_ms_and_wxyz(tmp_path):
    converter = import_convert_acp_raw()
    session_dir = tmp_path / "record_test"
    camera_dir = session_dir / "cam_test"
    write_fake_images(camera_dir, 2)
    np.save(
        camera_dir / "timestamps_host_s.npy",
        np.array([100.01, 100.06], dtype=np.float64),
    )
    write_common_arrays(session_dir)

    output_root = tmp_path / "acp_raw"
    episode_dir = converter.convert_session(
        session_dir=session_dir,
        output_root=output_root,
        episode_name="episode_000",
        camera_name="cam_test",
    )

    assert episode_dir == output_root / "episode_000"
    rgb_files = sorted((episode_dir / "rgb_0").iterdir())
    assert [p.name for p in rgb_files] == [
        "img_000000_00020.00000_ms.png",
        "img_000001_00070.00000_ms.png",
    ]
    assert rgb_files[0].read_bytes() == b"color-0"

    robot_records = json.loads((episode_dir / "robot_data_0.json").read_text())
    assert robot_records[0]["robot_time_stamps"] == 10.0
    assert robot_records[1]["robot_time_stamps"] == 60.0
    assert robot_records[0]["ts_pose_fb"] == [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]
    assert robot_records[0]["ts_pose_command"] == [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]

    wrench_records = json.loads((episode_dir / "wrench_data_0.json").read_text())
    assert wrench_records[0]["wrench_time_stamps"] == 0.0
    assert wrench_records[2]["wrench_time_stamps"] == 80.0
    assert wrench_records[1]["wrench"] == [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]


def test_convert_session_uses_robot_timestamps_for_legacy_camera_without_timestamps(tmp_path):
    converter = import_convert_acp_raw()
    session_dir = tmp_path / "record_legacy"
    camera_dir = session_dir / "cam_legacy"
    write_fake_images(camera_dir, 2)
    write_common_arrays(session_dir, legacy=True)

    episode_dir = converter.convert_session(
        session_dir=session_dir,
        output_root=tmp_path / "acp_raw",
        episode_name="episode_legacy",
        camera_name="cam_legacy",
    )

    rgb_files = sorted((episode_dir / "rgb_0").iterdir())
    assert [p.name for p in rgb_files] == [
        "img_000000_00010.00000_ms.png",
        "img_000001_00060.00000_ms.png",
    ]
