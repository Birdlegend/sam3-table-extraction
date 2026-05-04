#!/usr/bin/env python3
"""Post-hoc threshold analysis for table-detection eval runs.

Reads ``predictions.coco.json`` and ``ground_truth.coco.json`` from an eval
output dir produced by ``eval_tablebank.py`` / ``eval_pubtables.py`` and writes:

  - score_histogram.png   (TP vs FP score distributions)
  - f1_vs_threshold.png   (P / R / F1 across thresholds, with best F1 marked)
  - pr_curve.png          (PR curve, parametric on score threshold)
  - threshold_sweep.json  (full table of {threshold, p, r, f1, tp, fp, fn})
  - threshold_sweep.csv

NOTE: predictions.coco.json only contains predictions whose score was
already >= the eval-time ``score_threshold`` (default 0.25). The sweep can
therefore only sweep UPWARD from that floor; lower thresholds will give the
same numbers as the floor. To inspect lower thresholds, re-run eval with
``--score-threshold 0.0``.

Usage
-----
    modal run analyze_predictions.py \
        --eval-output-dir /artifacts/pubtables_eval_leaders/<study>/<split>/<run>
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


ANALYSIS_IMAGE = (
    modal.Image.debian_slim()
    .pip_install("matplotlib", "numpy")
)

app = modal.App(name="pubtables-eval-analysis", image=ANALYSIS_IMAGE)

artifacts_vol = modal.Volume.from_name("artifacts-vol")
MODAL_ARTIFACTS_DIR = "/artifacts"


def _bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _match_predictions(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int]:
    """Return ``(labeled_predictions, total_gt)``.

    ``labeled_predictions`` is a list of ``(score, is_tp)`` tuples sorted by
    score descending. Greedy per-image matching: each GT can match at most one
    prediction, predictions are processed highest-score-first.
    """
    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    labeled: list[tuple[float, bool]] = []
    all_image_ids = set(preds_by_image) | set(anns_by_image)
    for image_id in all_image_ids:
        preds = sorted(
            preds_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        gts = anns_by_image.get(image_id, [])
        matched_gt: set[int] = set()
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt:
                    continue
                iou = _bbox_iou_xywh(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            is_tp = best_iou >= iou_threshold and best_idx >= 0
            if is_tp:
                matched_gt.add(best_idx)
            labeled.append((float(pred.get("score", 0.0)), is_tp))

    labeled.sort(key=lambda x: x[0], reverse=True)
    total_gt = sum(len(v) for v in anns_by_image.values())
    return labeled, total_gt


def _compute_threshold_sweep(
    labeled_preds: list[tuple[float, bool]],
    total_gt: int,
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for threshold in thresholds:
        kept = [(s, tp) for (s, tp) in labeled_preds if s >= threshold]
        tp = sum(1 for (_, t) in kept if t)
        fp = sum(1 for (_, t) in kept if not t)
        fn = total_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return results


def _save_score_histogram(
    labeled_preds: list[tuple[float, bool]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp_scores = [s for (s, t) in labeled_preds if t]
    fp_scores = [s for (s, t) in labeled_preds if not t]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 50
    ax.hist(
        tp_scores,
        bins=bins,
        alpha=0.6,
        label=f"TP ({len(tp_scores):,})",
        color="#2ca02c",
    )
    ax.hist(
        fp_scores,
        bins=bins,
        alpha=0.6,
        label=f"FP ({len(fp_scores):,})",
        color="#d62728",
    )
    ax.set_xlabel("Prediction score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution: TP vs FP (IoU>=0.50)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_f1_vs_threshold(
    sweep_results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in sweep_results]
    p_vals = [r["precision"] for r in sweep_results]
    r_vals = [r["recall"] for r in sweep_results]
    f_vals = [r["f1"] for r in sweep_results]
    best = max(sweep_results, key=lambda row: row["f1"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, p_vals, label="Precision", linewidth=2)
    ax.plot(thresholds, r_vals, label="Recall", linewidth=2)
    ax.plot(thresholds, f_vals, label="F1", linewidth=2)
    ax.axvline(
        best["threshold"],
        linestyle="--",
        color="black",
        alpha=0.5,
        label=f"best F1 @ thr={best['threshold']:.3f} (F1={best['f1']:.4f})",
    )
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Precision / Recall / F1 vs score threshold (IoU>=0.50)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_pr_curve(
    sweep_results: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_vals = [r["precision"] for r in sweep_results]
    r_vals = [r["recall"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(r_vals, p_vals, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve @ IoU>=0.50 (parametric on score threshold)")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _analyze_run(
    eval_output_dir: str,
    iou_threshold: float,
    thresholds: list[float],
) -> dict[str, Any]:
    eval_dir = Path(eval_output_dir)
    predictions_path = eval_dir / "predictions.coco.json"
    gt_path = eval_dir / "ground_truth.coco.json"

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    predictions = json.loads(predictions_path.read_text())
    gt_payload = json.loads(gt_path.read_text())
    annotations = gt_payload["annotations"]

    print(
        f"[analyze] Loaded {len(predictions):,} predictions and "
        f"{len(annotations):,} GT annotations from {eval_dir}",
        flush=True,
    )

    labeled_preds, total_gt = _match_predictions(
        predictions, annotations, iou_threshold
    )
    inferred_floor = (
        min(s for (s, _) in labeled_preds) if labeled_preds else 0.0
    )
    print(
        f"[analyze] Matched {len(labeled_preds):,} predictions against "
        f"{total_gt:,} GTs at IoU>={iou_threshold:.2f}; "
        f"min observed score in predictions = {inferred_floor:.4f}",
        flush=True,
    )

    # Drop sweep thresholds below the eval-time floor; they are degenerate
    # because predictions below that floor were never written to disk.
    effective_thresholds = [t for t in thresholds if t >= inferred_floor - 1e-9]
    if not effective_thresholds:
        effective_thresholds = [inferred_floor]
    sweep = _compute_threshold_sweep(labeled_preds, total_gt, effective_thresholds)

    out_hist = eval_dir / "score_histogram.png"
    out_f1 = eval_dir / "f1_vs_threshold.png"
    out_pr = eval_dir / "pr_curve.png"
    out_json = eval_dir / "threshold_sweep.json"
    out_csv = eval_dir / "threshold_sweep.csv"

    _save_score_histogram(labeled_preds, out_hist)
    _save_f1_vs_threshold(sweep, out_f1)
    _save_pr_curve(sweep, out_pr)

    out_json.write_text(json.dumps(sweep, indent=2))
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        for row in sweep:
            writer.writerow(row)

    best = max(sweep, key=lambda row: row["f1"])
    summary = {
        "eval_output_dir": str(eval_dir),
        "iou_threshold": iou_threshold,
        "num_predictions": len(predictions),
        "num_gt_annotations": int(total_gt),
        "min_score_in_predictions": float(inferred_floor),
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "files": {
            "score_histogram": str(out_hist),
            "f1_vs_threshold": str(out_f1),
            "pr_curve": str(out_pr),
            "threshold_sweep_json": str(out_json),
            "threshold_sweep_csv": str(out_csv),
        },
        "note": (
            "Sweep is restricted to thresholds >= the eval-time score floor. "
            "Re-run eval with score_threshold=0.0 to inspect lower thresholds."
        ),
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[analyze] Wrote analysis files to {eval_dir}", flush=True)
    return summary


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 30,
)
def analyze_remote(
    eval_output_dir: str,
    iou_threshold: float = 0.5,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_steps: int = 101,
) -> dict[str, Any]:
    artifacts_vol.reload()
    span = max(2, int(threshold_steps))
    thresholds = [
        threshold_min + (threshold_max - threshold_min) * i / (span - 1)
        for i in range(span)
    ]
    result = _analyze_run(eval_output_dir, iou_threshold, thresholds)
    artifacts_vol.commit()
    return result


@app.local_entrypoint()
def main(
    eval_output_dir: str,
    iou_threshold: float = 0.5,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_steps: int = 101,
):
    """Run threshold analysis on an existing eval output directory.

    ``eval_output_dir`` should be the absolute path on the artifacts volume,
    e.g. ``/artifacts/pubtables_eval_leaders/sam3-final-optuna-asha/test/trial_0006_20260501-040000``.
    """
    result = analyze_remote.remote(
        eval_output_dir=eval_output_dir,
        iou_threshold=iou_threshold,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_steps=threshold_steps,
    )
    print(json.dumps(result, indent=2))
