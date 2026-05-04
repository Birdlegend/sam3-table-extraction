#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

TABLEBANK_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("triton", "pycocotools")
    .add_local_python_source("sam3_table")
    .add_local_file("voc_to_coco.py", remote_path="/root/voc_to_coco.py")
    .add_local_file("eval_tablebank.py", remote_path="/root/eval_tablebank.py")
)

DEFAULT_STUDY_NAME = "sam3-final-optuna-asha"
DEFAULT_NUM_RUNG_STAGES = 5

PUBTABLES_DATASET_SUBDIR = "pubtables_v2_single_pages"
SUPPORTED_SPLITS = ("train", "val", "test")


def _resolve_split_paths(split: str) -> tuple[str, str]:
    normalized_split = split.strip().lower()
    if normalized_split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of: {SUPPORTED_SPLITS}."
        )
    dataset_root = f"/data/{PUBTABLES_DATASET_SUBDIR}/{normalized_split}"
    annotations_path = f"{dataset_root}/_annotations.coco.json"
    return dataset_root, annotations_path


def _resolve_dataset_fraction(
    annotations_path: str,
    dataset_fraction: float,
    num_images: int,
) -> tuple[float, int]:
    payload = json.loads(Path(annotations_path).read_text())
    total_images = len(payload.get("images", []))
    if num_images <= 0 or total_images == 0:
        return float(dataset_fraction), total_images
    if num_images >= total_images:
        return 1.0, total_images
    resolved = float(num_images) / float(total_images)
    return min(max(resolved, 1e-6), 1.0), total_images


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _eval_tablebank_module():
    import importlib

    return importlib.import_module("eval_tablebank")


def _bind_eval_tablebank_runtime(eval_tablebank: Any) -> None:
    # Rebind volume handles so `.local()` calls run against volumes attached
    # to this app/function container instead of the original tablebank app.
    eval_tablebank.tablebank_vol = pubtables_vol
    eval_tablebank.artifacts_vol = artifacts_vol


def _resolve_table_category_ids(coco_payload: dict[str, Any]) -> set[int]:
    categories = coco_payload.get("categories", [])
    exact = {
        int(cat["id"])
        for cat in categories
        if str(cat.get("name", "")).strip().lower() == "table"
    }
    if exact:
        return exact

    # Fallback for slight naming variation while still excluding cell/row/column labels.
    fallback = {
        int(cat["id"])
        for cat in categories
        if "table" in str(cat.get("name", "")).strip().lower()
        and "cell" not in str(cat.get("name", "")).strip().lower()
        and "row" not in str(cat.get("name", "")).strip().lower()
        and "column" not in str(cat.get("name", "")).strip().lower()
    }
    if fallback:
        return fallback

    available = [str(cat.get("name", "")) for cat in categories]
    raise ValueError(
        "Could not identify a table category in annotations. "
        f"Available categories: {available}"
    )


def _write_table_only_annotations(
    source_annotations_path: str,
    output_dir: str,
) -> str:
    source_path = Path(source_annotations_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {source_path}")

    payload = json.loads(source_path.read_text())
    table_category_ids = _resolve_table_category_ids(payload)
    total_annotations = len(payload.get("annotations", []))
    total_images = len(payload.get("images", []))

    filtered_annotations = [
        ann
        for ann in payload.get("annotations", [])
        if int(ann.get("category_id", -1)) in table_category_ids
    ]
    image_ids_with_table = {int(ann["image_id"]) for ann in filtered_annotations}
    filtered_images = [
        image
        for image in payload.get("images", [])
        if int(image.get("id", -1)) in image_ids_with_table
    ]

    filtered_payload = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": [{"id": 1, "name": "table"}],
        "info": payload.get("info", {}),
    }
    for ann in filtered_payload["annotations"]:
        ann["category_id"] = 1

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / "pubtables_table_only_annotations.coco.json"
    filtered_path.write_text(json.dumps(filtered_payload, indent=2, default=_json_default))
    print(
        "[pubtables-eval] table-only GT filter: "
        f"kept {len(filtered_annotations):,}/{total_annotations:,} annotations "
        f"across {len(filtered_images):,}/{total_images:,} images "
        f"(table category ids={sorted(table_category_ids)}); "
        f"wrote {filtered_path}",
        flush=True,
    )
    return str(filtered_path)

app = modal.App(name="pubtables-eval", image=TABLEBANK_IMAGE)

pubtables_vol = modal.Volume.from_name("pubtables-vol")
artifacts_vol = modal.Volume.from_name("artifacts-vol", create_if_missing=True)

MODAL_DATA_DIR = "/data"
MODAL_ARTIFACTS_DIR = "/artifacts"


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def get_current_optuna_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
) -> dict[str, Any]:
    eval_tablebank = _eval_tablebank_module()
    _bind_eval_tablebank_runtime(eval_tablebank)
    return eval_tablebank.get_current_optuna_leader.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=num_rung_stages,
    )


@app.function(
    image=TABLEBANK_IMAGE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: pubtables_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_pubtables_eval_on_current_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    split: str = "val",
    dataset_root: str | None = None,
    annotations_path: str | None = None,
    output_root_dir: str = "/artifacts/pubtables_eval/sam3-final-optuna-asha/threshold_tuning",
    score_threshold: float = 0.25,
    query_text: str = "table",
    batch_size: int = 32,
    visualize_max_images: int = 20,
    dataset_fraction: float = 0.1,
    num_images: int = 0,
    sample_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
    num_workers: int = 8,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    pubtables_vol.reload()
    resolved_root, resolved_annotations = _resolve_split_paths(split)
    dataset_root = dataset_root or resolved_root
    annotations_path = annotations_path or resolved_annotations
    print(
        f"[pubtables-eval] split={split} dataset_root={dataset_root} "
        f"annotations_path={annotations_path}",
        flush=True,
    )
    leader = get_current_optuna_leader.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=num_rung_stages,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    study_slug = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in study_name)
    leader_output_dir = (
        f"{output_root_dir.rstrip('/')}"
        f"/{study_slug}/{split}/trial_{int(leader['trial_number']):04d}_{timestamp}"
    )
    result = run_pubtables_eval.remote(
        weights_path=str(leader["weights_path"]),
        dataset_root=dataset_root,
        annotations_path=annotations_path,
        output_dir=leader_output_dir,
        score_threshold=score_threshold,
        query_text=query_text,
        batch_size=batch_size,
        visualize_max_images=visualize_max_images,
        dataset_fraction=dataset_fraction,
        num_images=num_images,
        sample_seed=sample_seed,
        duplicate_iou_threshold=duplicate_iou_threshold,
        min_box_area=min_box_area,
        split=split,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    result["leader"] = leader
    result["benchmark_output_dir"] = leader_output_dir
    result["split"] = split
    return result


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def describe_current_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
) -> dict[str, Any]:
    eval_tablebank = _eval_tablebank_module()
    _bind_eval_tablebank_runtime(eval_tablebank)
    return eval_tablebank.describe_current_leader.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=num_rung_stages,
    )


@app.function(
    gpu="RTX-PRO-6000",
    image=TABLEBANK_IMAGE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: pubtables_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_pubtables_eval(
    weights_path: str,
    dataset_root: str,
    annotations_path: str,
    output_dir: str,
    score_threshold: float = 0.25,
    unique_output_dir: bool = False,
    dataset_fraction: float = 0.1,
    num_images: int = 0,
    sample_seed: int | None = None,
    holdout_train_fraction: float = 0.0,
    holdout_train_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    query_text: str = "table",
    batch_size: int = 8,
    visualize_max_images: int = 20,
    split: str = "val",
    num_workers: int = 8,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    eval_tablebank = _eval_tablebank_module()
    _bind_eval_tablebank_runtime(eval_tablebank)
    pubtables_vol.reload()
    print(
        f"[pubtables-eval] output directory: {output_dir}",
        flush=True,
    )
    resolved_fraction, total_images = _resolve_dataset_fraction(
        annotations_path=annotations_path,
        dataset_fraction=dataset_fraction,
        num_images=num_images,
    )
    target_images = max(1, int(total_images * resolved_fraction)) if total_images > 0 else 0
    print(
        f"[pubtables-eval] split={split} total_images_in_split={total_images:,} "
        f"requested_num_images={num_images} dataset_fraction={resolved_fraction:.6f} "
        f"approx_target_images={target_images:,}",
        flush=True,
    )
    table_only_annotations_path = _write_table_only_annotations(
        source_annotations_path=annotations_path,
        output_dir=output_dir,
    )
    return eval_tablebank.run_tablebank_eval.local(
        weights_path=weights_path,
        dataset_root=dataset_root,
        annotations_path=table_only_annotations_path,
        output_dir=output_dir,
        score_threshold=score_threshold,
        unique_output_dir=unique_output_dir,
        dataset_fraction=resolved_fraction,
        sample_seed=sample_seed,
        holdout_train_fraction=holdout_train_fraction,
        holdout_train_seed=holdout_train_seed,
        duplicate_iou_threshold=duplicate_iou_threshold,
        min_box_area=min_box_area,
        query_text=query_text,
        batch_size=batch_size,
        visualize_max_images=visualize_max_images,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


@app.local_entrypoint()
def main(
    weights: str = "",
    split: str = "test",
    dataset_root: str = "",
    annotations: str = "",
    output_dir: str = "/artifacts/pubtables_eval",
    score_threshold: float = 0.25,
    unique_output_dir: bool = False,
    dataset_fraction: float = 0.1,
    num_images: int = 5000,
    sample_seed: int | None = None,
    holdout_train_fraction: float = 0.0,
    holdout_train_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    query_text: str = "table",
    batch_size: int = 32,
    visualize_max_images: int = 20,
    use_current_leader: bool = False,
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
    show_current_leader_only: bool = False,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    resolved_dataset_root, resolved_annotations = _resolve_split_paths(split)
    dataset_root = dataset_root or resolved_dataset_root
    annotations = annotations or resolved_annotations

    if show_current_leader_only:
        result = describe_current_leader.remote(
            study_name=study_name,
            include_running_trials=include_running_trials,
            sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
            num_rung_stages=num_rung_stages,
        )
    elif use_current_leader:
        result = run_pubtables_eval_on_current_leader.remote(
            study_name=study_name,
            split=split,
            dataset_root=dataset_root,
            annotations_path=annotations,
            output_root_dir=output_dir,
            score_threshold=score_threshold,
            query_text=query_text,
            batch_size=batch_size,
            visualize_max_images=visualize_max_images,
            dataset_fraction=dataset_fraction,
            num_images=num_images,
            sample_seed=sample_seed,
            duplicate_iou_threshold=duplicate_iou_threshold,
            min_box_area=min_box_area,
            include_running_trials=include_running_trials,
            sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
            num_rung_stages=num_rung_stages,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        if not weights:
            raise ValueError(
                "weights is required unless use_current_leader=True."
            )
        result = run_pubtables_eval.remote(
            weights_path=weights,
            dataset_root=dataset_root,
            annotations_path=annotations,
            output_dir=output_dir,
            score_threshold=score_threshold,
            unique_output_dir=unique_output_dir,
            dataset_fraction=dataset_fraction,
            num_images=num_images,
            sample_seed=sample_seed,
            holdout_train_fraction=holdout_train_fraction,
            holdout_train_seed=holdout_train_seed,
            duplicate_iou_threshold=duplicate_iou_threshold,
            min_box_area=min_box_area,
            query_text=query_text,
            batch_size=batch_size,
            visualize_max_images=visualize_max_images,
            split=split,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    print(json.dumps(result, indent=2, default=_json_default))
