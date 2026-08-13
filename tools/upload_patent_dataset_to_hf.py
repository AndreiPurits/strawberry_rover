#!/usr/bin/env python3
"""Upload patent images, labels, and frozen weights to HF dataset AndreiPurits/strawBerry.

Uses hardlinks when possible (Orin has little free disk). Prefers huggingface_hub;
falls back to git+SSH if the hub library is missing.

SSH: Host hf.co must use ~/.ssh/id_ed25519_github, and that public key must be
registered at https://huggingface.co/settings/keys
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/andrei/project/strawberry_rover_ws")
HF_REPO_ID = "AndreiPurits/strawBerry"
HF_GIT = "git@hf.co:datasets/AndreiPurits/strawBerry"
STAGING = Path("/home/andrei/project/.hf_strawberry_staging")

# (local_path, remote_path) — images + labels, no derived bbox_vis/preview.
DATASETS = [
    (REPO / "data/final_detection_dataset/images", "images/detection_final/images"),
    (REPO / "data/final_detection_dataset/labels", "images/detection_final/labels"),
    (REPO / "data/yolo_detection_dataset_v3", "images/detection_v3"),
    (REPO / "data/classification_dataset_v2", "images/classification_v2"),
    (REPO / "data/classification_manual", "images/classification_manual"),
    (REPO / "data/yolo_segmentation_dataset", "images/segmentation_yolo"),
    (REPO / "data/segmentation_project_dataset", "images/segmentation_project"),
    (REPO / "data/плодоножки апрувд", "images/peduncle_approved"),
    (REPO / "data/peduncle_obb_berry_anchored_v21", "images/peduncle_obb_v21"),
    (REPO / "data/ФПС ДАТАСЕТ", "images/fps_holdout"),
]

MODELS = [
    (
        REPO / "runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt",
        "models/detector_yolov8s_v3_lowdensity_best.pt",
    ),
    (
        REPO / "runs/classification_benchmark_v2/efficientnet_b0/best.pt",
        "models/classifier_efficientnet_b0_best.pt",
    ),
    (
        REPO / "runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt",
        "models/segmenter_yolov8n_seg_best.pt",
    ),
    (
        REPO / "runs/peduncle_obb/yolov8n_peduncle_v21_mask_calyx/weights/best.pt",
        "models/peduncle_obb_yolov8n_v21_best.pt",
    ),
]

SKIP_DIR_NAMES = {
    "bbox_vis",
    "preview",
    "crops_preview",
    "reports",
    "__pycache__",
    "reserve",
    "duplicates_removed",
    "rejected",
    "rejected_no_strawberry",
    "rejected_no_peduncle",
}


def hardlink_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.exists():
        print(f"SKIP missing {src}")
        return 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            t = target_dir / name
            if t.exists():
                continue
            try:
                os.link(s, t)
            except OSError:
                shutil.copy2(s, t)
            n += 1
    return n


def write_dataset_card(dst: Path) -> None:
    (dst / "README.md").write_text(
        """---
license: other
pretty_name: strawBerry patent dump
---

# strawBerry

Private dump of labeled strawberry frames and frozen on-rover weights for the
Strawberry Rover patent package.

Companion code: https://github.com/AndreiPurits/strawberry_rover/tree/Патент

See `PATENT.md` in that branch for task-by-task layout.
""",
        encoding="utf-8",
    )


def try_hub_upload(staging: Path) -> bool:
    try:
        from huggingface_hub import HfApi, HfFolder
    except ImportError:
        return False
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    if not token:
        print("No HF_TOKEN / huggingface-cli login; hub upload skipped.")
        return False
    api = HfApi(token=token)
    print("Uploading via huggingface_hub.upload_folder ...")
    # 0.23 has upload_folder only (no upload_large_folder).
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=str(staging),
        repo_type="dataset",
        commit_message="Add labeled frames and frozen production weights",
    )
    return True


def git_push(staging: Path) -> None:
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        "ssh -i /home/andrei/.ssh/id_ed25519_github -o IdentitiesOnly=yes"
    )
    if not (staging / ".git").exists():
        subprocess.check_call(["git", "init"], cwd=staging, env=env)
        subprocess.check_call(["git", "remote", "add", "origin", HF_GIT], cwd=staging, env=env)
    subprocess.check_call(["git", "add", "-A"], cwd=staging, env=env)
    subprocess.check_call(
        ["git", "commit", "-m", "Add labeled frames and frozen production weights"],
        cwd=staging,
        env=env,
    )
    subprocess.check_call(["git", "branch", "-M", "main"], cwd=staging, env=env)
    subprocess.check_call(["git", "push", "-u", "origin", "main"], cwd=staging, env=env)


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    write_dataset_card(STAGING)
    total = 0
    for src, rel in DATASETS:
        print(f"link {src} -> {rel}")
        total += hardlink_tree(src, STAGING / rel)
    for src, rel in MODELS:
        dst = STAGING / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"SKIP missing model {src}")
            continue
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        total += 1
        print(f"model {rel}")
    print(f"staged files: {total}")
    if try_hub_upload(STAGING):
        print("Hub upload done:", f"https://huggingface.co/datasets/{HF_REPO_ID}")
        return 0
    print("huggingface_hub missing or failed; trying git push ...")
    git_push(STAGING)
    print("git push done:", f"https://huggingface.co/datasets/{HF_REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
