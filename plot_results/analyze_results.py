#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from mizani.formatters import percent_format

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_wrap,
    geom_col,
    geom_point,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)


FAMILY_ORDER = {"llama": 0, "gemma": 1}
VARIANT_ORDER = {"0-shot": 0, "4-shot": 1, "SFT": 2, "DPO": 3}
DEFAULT_LABEL_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
PRIMARY_BEHAVIOR_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
CLASS_DISPLAY = {
    "direct": "direct",
    "tool_call": "tool_call",
    "request_for_info": "request_for_info",
    "cannot_answer": "cannot_answer",
}
OUTCOME_ORDER = ["stay_correct", "fixed", "broken", "stay_wrong"]
OUTCOME_COLORS = {
    "stay_correct": "#2f9e44",
    "fixed": "#1971c2",
    "broken": "#c92a2a",
    "stay_wrong": "#868e96",
}
SCORING_COLORS = {
    "raw": "#6c757d",
    "normalized": "#0b7285",
}
FAMILY_COLORS = {
    "llama": "#1f78b4",
    "gemma": "#e67e22",
}
PREDICTION_COLORS = {
    "tool_call": "#1f78b4",
    "request_for_info": "#2ca02c",
    "cannot_answer": "#d62728",
    "direct": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_runs_dir = repo_root / "Prompting Baselines"
    default_output_dir = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser(
        description="Analyze prompting baseline results and generate ggplot charts + summary tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", default=str(default_runs_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--top-sources", type=int, default=8)
    parser.add_argument("--min-source-examples", type=int, default=100)
    parser.add_argument(
        "--include-direct-class",
        action="store_true",
        help="Include direct in class-level heatmap (usually has zero support in this split).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_run_dirs(runs_dir: Path) -> List[Path]:
    required = {"summary.json", "run_config.json"}
    run_dirs: List[Path] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        child_files = {p.name for p in child.iterdir() if p.is_file()}
        if required.issubset(child_files):
            run_dirs.append(child)
    return sorted(run_dirs)


def infer_model_family(run_key: str, run_config: Dict[str, Any]) -> str:
    family = str(run_config.get("model_family", "")).lower().strip()
    model_name = str(run_config.get("model_name_or_path", "")).lower()
    haystack = f"{run_key.lower()} {family} {model_name}"
    if "llama" in haystack:
        return "llama"
    if "gemma" in haystack:
        return "gemma"
    return family or "unknown"


def infer_variant_label(run_key: str, run_config: Dict[str, Any]) -> str:
    haystack = " ".join(
        [
            run_key.lower(),
            str(run_config.get("output_dir", "")).lower(),
            str(run_config.get("model_name_or_path", "")).lower(),
        ]
    )
    num_shots = int(run_config.get("num_shots", 0) or 0)
    if "dpo" in haystack:
        return "DPO"
    if "sft" in haystack:
        return "SFT"
    if "4shot" in haystack or num_shots == 4:
        return "4-shot"
    if num_shots == 0:
        return "0-shot"
    return f"{num_shots}-shot"


def family_display_names(model_family: str) -> Tuple[str, str]:
    if model_family == "llama":
        return "Llama 3.2 3B", "Llama"
    if model_family == "gemma":
        return "Gemma 3 4B", "Gemma"
    return model_family, model_family


def extract_class_metrics(
    *,
    run_key: str,
    run_display: str,
    model_family: str,
    variant_label: str,
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    label_order = list(summary.get("label_order") or DEFAULT_LABEL_ORDER)
    records: List[Dict[str, Any]] = []
    for scoring in ("raw", "normalized"):
        class_report = summary.get(scoring, {}).get("classification_report", {})
        cm = summary.get(scoring, {}).get("confusion_matrix", [])
        if not cm:
            continue
        total = float(sum(sum(int(x) for x in row) for row in cm))
        row_sums = [float(sum(int(x) for x in row)) for row in cm]
        col_sums = [float(sum(int(cm[r][c]) for r in range(len(cm)))) for c in range(len(cm))]
        for label in label_order:
            if label not in class_report:
                continue
            metrics = class_report[label]
            idx = label_order.index(label)
            tp = float(cm[idx][idx]) if idx < len(cm) and idx < len(cm[idx]) else 0.0
            fn = row_sums[idx] - tp if idx < len(row_sums) else 0.0
            fp = col_sums[idx] - tp if idx < len(col_sums) else 0.0
            tn = total - tp - fn - fp
            one_vs_rest_accuracy = (tp + tn) / total if total > 0 else 0.0
            records.append(
                {
                    "run_key": run_key,
                    "run_display": run_display,
                    "model_family": model_family,
                    "variant_label": variant_label,
                    "scoring": scoring,
                    "behavior_class": label,
                    "accuracy": one_vs_rest_accuracy,
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1-score", 0.0)),
                    "support": float(metrics.get("support", 0.0)),
                }
            )
    return records


def normalization_outcome(gold: str, pred_raw: str, pred_norm: str) -> str:
    raw_correct = pred_raw == gold
    norm_correct = pred_norm == gold
    if raw_correct and norm_correct:
        return "stay_correct"
    if (not raw_correct) and norm_correct:
        return "fixed"
    if raw_correct and (not norm_correct):
        return "broken"
    return "stay_wrong"


def extract_sample_level_summaries(
    *,
    run_key: str,
    run_display: str,
    model_family: str,
    variant_label: str,
    samples: List[Dict[str, Any]],
    label_order: Iterable[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not samples:
        return [], [], [], []

    n_examples = len(samples)
    direct_rate_records: List[Dict[str, Any]] = []
    outcome_records: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    prediction_mix_records: List[Dict[str, Any]] = []

    for scoring, pred_col in (("raw", "pred_raw"), ("normalized", "pred_norm")):
        direct_predictions = sum(1 for row in samples if row.get(pred_col) == "direct")
        direct_rate_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "variant_label": variant_label,
                "scoring": scoring,
                "n_examples": n_examples,
                "direct_predictions": direct_predictions,
                "direct_prediction_rate": direct_predictions / max(n_examples, 1),
            }
        )

        counts: Dict[str, int] = {}
        for row in samples:
            label = str(row.get(pred_col))
            counts[label] = counts.get(label, 0) + 1
        for label in label_order:
            prediction_mix_records.append(
                {
                    "run_key": run_key,
                    "run_display": run_display,
                    "model_family": model_family,
                    "variant_label": variant_label,
                    "scoring": scoring,
                    "label": label,
                    "fraction": counts.get(label, 0) / max(n_examples, 1),
                }
            )

    outcome_counts = {name: 0 for name in OUTCOME_ORDER}
    source_stats: Dict[str, Dict[str, float]] = {}
    for row in samples:
        gold = str(row.get("gold"))
        pred_raw = str(row.get("pred_raw"))
        pred_norm = str(row.get("pred_norm"))
        outcome = normalization_outcome(gold, pred_raw, pred_norm)
        outcome_counts[outcome] += 1

        source = str(row.get("source") or "unknown")
        stats = source_stats.setdefault(
            source,
            {"n_examples": 0, "raw_correct": 0, "normalized_correct": 0},
        )
        stats["n_examples"] += 1
        stats["raw_correct"] += 1 if pred_raw == gold else 0
        stats["normalized_correct"] += 1 if pred_norm == gold else 0

    for outcome_name, count in outcome_counts.items():
        outcome_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "variant_label": variant_label,
                "outcome": outcome_name,
                "count": count,
                "fraction": count / max(n_examples, 1),
            }
        )

    for source, stats in source_stats.items():
        n_source = int(stats["n_examples"])
        source_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "variant_label": variant_label,
                "source": source,
                "n_examples": n_source,
                "raw_accuracy": stats["raw_correct"] / max(n_source, 1),
                "normalized_accuracy": stats["normalized_correct"] / max(n_source, 1),
            }
        )

    return direct_rate_records, outcome_records, source_records, prediction_mix_records


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["family_rank"] = frame["model_family"].map(FAMILY_ORDER).fillna(99).astype(int)
    frame["variant_rank"] = frame["variant_label"].map(VARIANT_ORDER).fillna(99).astype(int)
    frame = frame.sort_values(["family_rank", "variant_rank", "run_key"]).reset_index(drop=True)
    return frame


def apply_run_order(df: pd.DataFrame, run_order: List[str]) -> pd.DataFrame:
    frame = df.copy()
    frame["run_display"] = pd.Categorical(frame["run_display"], categories=run_order, ordered=True)
    return frame


def save_plot(plot_obj: Any, output_path: Path, width: float, height: float) -> None:
    plot_obj.save(
        filename=str(output_path),
        width=width,
        height=height,
        dpi=300,
        units="in",
        verbose=False,
    )


def save_plot_multi(
    plot_obj: Any,
    *,
    figures_dir: Path,
    base_name: str,
    width: float,
    height: float,
) -> None:
    save_plot(plot_obj, figures_dir / f"{base_name}.png", width=width, height=height)
    # PDF output keeps vector geometry for report-quality scaling.
    save_plot(plot_obj, figures_dir / f"{base_name}.pdf", width=width, height=height)


def remove_stale_figures(figures_dir: Path) -> None:
    stale_basenames = [
        "accuracy_raw_vs_normalized_by_run",
        "macro_f1_raw_vs_normalized_by_run",
        "normalized_recall_heatmap_by_class",
    ]
    for base in stale_basenames:
        for ext in ("png", "pdf"):
            path = figures_dir / f"{base}.{ext}"
            if path.exists():
                path.unlink()


def make_summary_charts(
    *,
    runs_df: pd.DataFrame,
    class_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    source_df: pd.DataFrame,
    prediction_mix_df: pd.DataFrame,
    figures_dir: Path,
    top_sources: int,
    min_source_examples: int,
    include_direct_class: bool,
) -> None:
    run_order = runs_df["run_display"].tolist()
    runs_plot = apply_run_order(runs_df, run_order)
    runs_plot["is_probe"] = runs_plot["variant_label"].str.contains("probe", case=False, na=False) | runs_plot[
        "run_key"
    ].str.contains("probe", case=False, na=False)

    runs_plot["model_family"] = runs_plot["model_family"].astype(str)
    runs_plot["norm_accuracy_label"] = runs_plot["norm_accuracy"].map(lambda x: f"{x:.3f}")
    runs_plot["norm_macro_f1_label"] = runs_plot["norm_macro_f1"].map(lambda x: f"{x:.3f}")
    runs_plot["norm_accuracy_label_pos"] = (runs_plot["norm_accuracy"] + 0.012).clip(upper=0.985)
    runs_plot["norm_macro_f1_label_pos"] = (runs_plot["norm_macro_f1"] + 0.012).clip(upper=0.985)

    accuracy_plot = (
        ggplot(runs_plot, aes(x="run_display", y="norm_accuracy", fill="model_family"))
        + geom_col(width=0.72)
        + geom_text(aes(y="norm_accuracy_label_pos", label="norm_accuracy_label"), size=10, ha="left")
        + coord_flip()
        + scale_fill_manual(values=FAMILY_COLORS)
        + scale_y_continuous(
            labels=percent_format(),
            limits=(0.0, 1.0),
            breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        + labs(
            title="Normalized Accuracy by Run",
            x="",
            y="Accuracy",
            fill="Model Family",
        )
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(
        accuracy_plot,
        figures_dir=figures_dir,
        base_name="normalized_accuracy_by_run",
        width=11,
        height=6,
    )

    macrof1_plot = (
        ggplot(runs_plot, aes(x="run_display", y="norm_macro_f1", fill="model_family"))
        + geom_col(width=0.72)
        + geom_text(aes(y="norm_macro_f1_label_pos", label="norm_macro_f1_label"), size=10, ha="left")
        + coord_flip()
        + scale_fill_manual(values=FAMILY_COLORS)
        + scale_y_continuous(
            labels=percent_format(),
            limits=(0.0, 1.0),
            breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        + labs(
            title="Normalized Macro-F1 by Run",
            x="",
            y="Macro-F1",
            fill="Model Family",
        )
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(
        macrof1_plot,
        figures_dir=figures_dir,
        base_name="normalized_macro_f1_by_run",
        width=11,
        height=6,
    )

    class_plot_df = class_df[class_df["scoring"] == "normalized"].copy()
    if not include_direct_class:
        class_plot_df = class_plot_df[class_plot_df["behavior_class"].isin(PRIMARY_BEHAVIOR_CLASSES)]
    class_plot_df = apply_run_order(class_plot_df, run_order)
    class_plot_df["behavior_class"] = class_plot_df["behavior_class"].map(CLASS_DISPLAY)
    metric_specs = [
        ("accuracy", "Per-Class Accuracy (One-vs-Rest)", "per_class_accuracy_heatmap_by_run"),
        ("precision", "Per-Class Precision", "per_class_precision_heatmap_by_run"),
        ("recall", "Per-Class Recall", "per_class_recall_heatmap_by_run"),
        ("f1", "Per-Class F1", "per_class_f1_heatmap_by_run"),
    ]
    for metric_col, metric_title, metric_file in metric_specs:
        metric_df = class_plot_df.copy()
        metric_df["label"] = metric_df[metric_col].map(lambda x: f"{x:.2f}")
        metric_plot = (
            ggplot(metric_df, aes(x="behavior_class", y="run_display", fill=metric_col))
            + geom_tile(color="white")
            + geom_text(aes(label="label"), size=7)
            + scale_fill_gradient(low="#e8f6f3", high="#0b7285", limits=(0.0, 1.0))
            + labs(
                title=metric_title,
                x="Class",
                y="",
                fill=metric_col.capitalize(),
            )
            + theme_bw()
            + theme(figure_size=(10, 6), axis_text_y=element_text(size=9))
        )
        save_plot_multi(
            metric_plot,
            figures_dir=figures_dir,
            base_name=metric_file,
            width=10,
            height=6,
        )

    direct_plot_df = apply_run_order(direct_df, run_order)
    direct_plot_df["label"] = direct_plot_df["direct_prediction_rate"].map(lambda x: f"{x:.1%}")
    direct_rate_plot = (
        ggplot(
            direct_plot_df,
            aes(x="run_display", y="direct_prediction_rate", fill="scoring"),
        )
        + geom_col(position=position_dodge(width=0.75), width=0.68)
        + coord_flip()
        + scale_fill_manual(values=SCORING_COLORS)
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="Unsupported 'direct' Prediction Rate",
            x="",
            y="Rate",
            fill="Scoring",
        )
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(
        direct_rate_plot,
        figures_dir=figures_dir,
        base_name="unsupported_direct_prediction_rate_by_run",
        width=11,
        height=6,
    )

    outcome_plot_df = apply_run_order(outcome_df, run_order)
    outcome_plot = (
        ggplot(outcome_plot_df, aes(x="run_display", y="fraction", fill="outcome"))
        + geom_col(width=0.75)
        + coord_flip()
        + scale_fill_manual(values=OUTCOME_COLORS)
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="Normalization Outcome Breakdown",
            x="",
            y="Fraction of Examples",
            fill="Outcome",
        )
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(
        outcome_plot,
        figures_dir=figures_dir,
        base_name="normalization_outcome_breakdown_by_run",
        width=11,
        height=6,
    )

    source_plot_df = source_df[source_df["n_examples"] >= min_source_examples].copy()
    if not source_plot_df.empty:
        top_source_list = (
            source_plot_df.groupby("source", as_index=False)["n_examples"]
            .mean()
            .sort_values("n_examples", ascending=False)
            .head(top_sources)["source"]
            .tolist()
        )
        source_plot_df = source_plot_df[source_plot_df["source"].isin(top_source_list)].copy()
        source_plot_df = apply_run_order(source_plot_df, run_order)
        source_plot_df["source"] = pd.Categorical(source_plot_df["source"], categories=top_source_list, ordered=True)
        source_plot_df["label"] = source_plot_df["normalized_accuracy"].map(lambda x: f"{x:.2f}")

        source_heatmap = (
            ggplot(source_plot_df, aes(x="source", y="run_display", fill="normalized_accuracy"))
            + geom_tile(color="white")
            + geom_text(aes(label="label"), size=6)
            + labs(
                title="Top Sources: Normalized Accuracy by Run",
                x="Source",
                y="",
                fill="Norm Accuracy",
            )
            + theme_bw()
            + theme(
                figure_size=(12, 7),
                axis_text_x=element_text(rotation=30, ha="right"),
                axis_text_y=element_text(size=9),
            )
        )
        save_plot_multi(
            source_heatmap,
            figures_dir=figures_dir,
            base_name="normalized_accuracy_heatmap_top_sources",
            width=12,
            height=7,
        )

    pred_mix_plot_df = prediction_mix_df[prediction_mix_df["scoring"] == "normalized"].copy()
    pred_mix_plot_df = apply_run_order(pred_mix_plot_df, run_order)
    pred_mix_plot = (
        ggplot(pred_mix_plot_df, aes(x="run_display", y="fraction", fill="label"))
        + geom_col(width=0.75)
        + coord_flip()
        + scale_fill_manual(values=PREDICTION_COLORS)
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="Normalized Prediction Mix by Run",
            x="",
            y="Fraction of Predictions",
            fill="Predicted Label",
        )
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(
        pred_mix_plot,
        figures_dir=figures_dir,
        base_name="normalized_prediction_mix_by_run",
        width=11,
        height=6,
    )

    # Additional variants without probe runs (requested), for key comparison plots.
    no_probe_runs = runs_plot[~runs_plot["is_probe"]].copy()
    if not no_probe_runs.empty:
        no_probe_order = no_probe_runs["run_display"].tolist()
        no_probe_runs["norm_accuracy_label_pos"] = (no_probe_runs["norm_accuracy"] + 0.012).clip(upper=0.985)
        no_probe_runs["norm_macro_f1_label_pos"] = (no_probe_runs["norm_macro_f1"] + 0.012).clip(upper=0.985)
        no_probe_acc_plot = (
            ggplot(no_probe_runs, aes(x="run_display", y="norm_accuracy", fill="model_family"))
            + geom_col(width=0.72)
            + geom_text(aes(y="norm_accuracy_label_pos", label="norm_accuracy_label"), size=10, ha="left")
            + coord_flip()
            + scale_fill_manual(values=FAMILY_COLORS)
            + scale_y_continuous(
                labels=percent_format(),
                limits=(0.0, 1.0),
                breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
            + labs(
                title="Normalized Accuracy by Run",
                x="",
                y="Accuracy",
                fill="Model Family",
            )
            + theme_bw()
            + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
        )
        save_plot_multi(
            no_probe_acc_plot,
            figures_dir=figures_dir,
            base_name="normalized_accuracy_by_run_without_probe",
            width=11,
            height=6,
        )

        no_probe_f1_plot = (
            ggplot(no_probe_runs, aes(x="run_display", y="norm_macro_f1", fill="model_family"))
            + geom_col(width=0.72)
            + geom_text(aes(y="norm_macro_f1_label_pos", label="norm_macro_f1_label"), size=10, ha="left")
            + coord_flip()
            + scale_fill_manual(values=FAMILY_COLORS)
            + scale_y_continuous(
                labels=percent_format(),
                limits=(0.0, 1.0),
                breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
            + labs(
                title="Normalized Macro-F1 by Run",
                x="",
                y="Macro-F1",
                fill="Model Family",
            )
            + theme_bw()
            + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
        )
        save_plot_multi(
            no_probe_f1_plot,
            figures_dir=figures_dir,
            base_name="normalized_macro_f1_by_run_without_probe",
            width=11,
            height=6,
        )

        no_probe_mix_df = prediction_mix_df[prediction_mix_df["scoring"] == "normalized"].copy()
        no_probe_mix_df = no_probe_mix_df[
            ~(
                no_probe_mix_df["variant_label"].str.contains("probe", case=False, na=False)
                | no_probe_mix_df["run_key"].str.contains("probe", case=False, na=False)
            )
        ].copy()
        no_probe_mix_df = apply_run_order(no_probe_mix_df, no_probe_order)
        no_probe_mix_plot = (
            ggplot(no_probe_mix_df, aes(x="run_display", y="fraction", fill="label"))
            + geom_col(width=0.75)
            + coord_flip()
            + scale_fill_manual(values=PREDICTION_COLORS)
            + scale_y_continuous(labels=percent_format())
            + labs(
                title="Normalized Prediction Mix by Run",
                x="",
                y="Fraction of Predictions",
                fill="Predicted Label",
            )
            + theme_bw()
            + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
        )
        save_plot_multi(
            no_probe_mix_plot,
            figures_dir=figures_dir,
            base_name="normalized_prediction_mix_by_run_without_probe",
            width=11,
            height=6,
        )


def write_summary_markdown(
    *,
    runs_df: pd.DataFrame,
    class_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    summary_path: Path,
) -> None:
    best_norm_acc = runs_df.loc[runs_df["norm_accuracy"].idxmax()]
    best_norm_f1 = runs_df.loc[runs_df["norm_macro_f1"].idxmax()]
    biggest_acc_gain = runs_df.loc[(runs_df["norm_accuracy"] - runs_df["raw_accuracy"]).idxmax()]
    biggest_f1_gain = runs_df.loc[(runs_df["norm_macro_f1"] - runs_df["raw_macro_f1"]).idxmax()]
    highest_direct_norm = direct_df[direct_df["scoring"] == "normalized"].sort_values(
        "direct_prediction_rate", ascending=False
    ).iloc[0]

    class_recall = class_df[
        (class_df["scoring"] == "normalized") & (class_df["behavior_class"].isin(PRIMARY_BEHAVIOR_CLASSES))
    ].copy()
    mean_recall = class_recall.groupby("behavior_class", as_index=False)["recall"].mean()
    hardest_class = mean_recall.sort_values("recall", ascending=True).iloc[0]

    lines = [
        "# Result Analysis Summary",
        "",
        f"- Best normalized accuracy: **{best_norm_acc['run_display']}** ({best_norm_acc['norm_accuracy']:.3f})",
        f"- Best normalized macro-F1: **{best_norm_f1['run_display']}** ({best_norm_f1['norm_macro_f1']:.3f})",
        (
            "- Largest normalization accuracy gain: "
            f"**{biggest_acc_gain['run_display']}** "
            f"({(biggest_acc_gain['norm_accuracy'] - biggest_acc_gain['raw_accuracy']):+.3f})"
        ),
        (
            "- Largest normalization macro-F1 gain: "
            f"**{biggest_f1_gain['run_display']}** "
            f"({(biggest_f1_gain['norm_macro_f1'] - biggest_f1_gain['raw_macro_f1']):+.3f})"
        ),
        (
            "- Highest unsupported `direct` prediction rate under normalized scoring: "
            f"**{highest_direct_norm['run_display']}** "
            f"({highest_direct_norm['direct_prediction_rate']:.1%})"
        ),
        (
            "- Hardest class on average (normalized recall): "
            f"**{hardest_class['behavior_class']}** ({hardest_class['recall']:.3f})"
        ),
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_raw_vs_normalized_latex_table(
    *,
    runs_df: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = runs_df.copy()
    ordered["delta_accuracy"] = ordered["norm_accuracy"] - ordered["raw_accuracy"]
    ordered["delta_macro_f1"] = ordered["norm_macro_f1"] - ordered["raw_macro_f1"]

    lines: List[str] = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Run & Raw Acc & Norm Acc & $\\Delta$ Acc & Raw Macro-F1 & Norm Macro-F1 & $\\Delta$ Macro-F1 \\\\",
        "    \\midrule",
    ]
    for _, row in ordered.iterrows():
        run_name = latex_escape(str(row["run_display"]))
        lines.append(
            "    "
            + f"{run_name} & "
            + f"{row['raw_accuracy']:.3f} & {row['norm_accuracy']:.3f} & {row['delta_accuracy']:+.3f} & "
            + f"{row['raw_macro_f1']:.3f} & {row['norm_macro_f1']:.3f} & {row['delta_macro_f1']:+.3f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Raw vs. normalized scoring comparison by run.}",
            "  \\label{tab:raw-vs-normalized-scoring}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    data_dir = ensure_dir(output_dir / "data")
    figures_dir = ensure_dir(output_dir / "figures")
    remove_stale_figures(figures_dir)

    run_dirs = discover_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {runs_dir} with summary.json + run_config.json."
        )

    run_records: List[Dict[str, Any]] = []
    class_records: List[Dict[str, Any]] = []
    direct_records: List[Dict[str, Any]] = []
    outcome_records: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    prediction_mix_records: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_key = run_dir.name
        summary = load_json(run_dir / "summary.json")
        run_config = load_json(run_dir / "run_config.json")

        model_family = infer_model_family(run_key, run_config)
        variant_label = infer_variant_label(run_key, run_config)
        _, family_short = family_display_names(model_family)
        run_display = f"{family_short} {variant_label} ({run_key})"

        run_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "variant_label": variant_label,
                "raw_accuracy": float(summary["raw"]["accuracy"]),
                "norm_accuracy": float(summary["normalized"]["accuracy"]),
                "raw_macro_f1": float(summary["raw"]["macro_f1"]),
                "norm_macro_f1": float(summary["normalized"]["macro_f1"]),
                "n_examples": int(summary.get("n_examples", 0)),
            }
        )
        class_records.extend(
            extract_class_metrics(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                variant_label=variant_label,
                summary=summary,
            )
        )

        sample_path = run_dir / "samples.jsonl"
        if sample_path.exists():
            samples = read_jsonl(sample_path)
            label_order = summary.get("label_order") or DEFAULT_LABEL_ORDER
            direct_rows, outcome_rows, source_rows, pred_mix_rows = extract_sample_level_summaries(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                variant_label=variant_label,
                samples=samples,
                label_order=label_order,
            )
            direct_records.extend(direct_rows)
            outcome_records.extend(outcome_rows)
            source_records.extend(source_rows)
            prediction_mix_records.extend(pred_mix_rows)

    runs_df = sort_runs(pd.DataFrame(run_records))
    class_df = sort_runs(pd.DataFrame(class_records))
    direct_df = sort_runs(pd.DataFrame(direct_records))
    outcome_df = sort_runs(pd.DataFrame(outcome_records))
    source_df = sort_runs(pd.DataFrame(source_records))
    prediction_mix_df = sort_runs(pd.DataFrame(prediction_mix_records))

    runs_df.to_csv(data_dir / "run_level_metrics.csv", index=False)
    class_df.to_csv(data_dir / "class_level_metrics.csv", index=False)
    direct_df.to_csv(data_dir / "direct_prediction_rate.csv", index=False)
    outcome_df.to_csv(data_dir / "normalization_outcomes.csv", index=False)
    source_df.to_csv(data_dir / "source_level_accuracy.csv", index=False)
    prediction_mix_df.to_csv(data_dir / "prediction_mix.csv", index=False)
    write_raw_vs_normalized_latex_table(
        runs_df=runs_df,
        output_path=output_dir / "raw_vs_normalized_change_table_latex.txt",
    )

    make_summary_charts(
        runs_df=runs_df,
        class_df=class_df,
        direct_df=direct_df,
        outcome_df=outcome_df,
        source_df=source_df,
        prediction_mix_df=prediction_mix_df,
        figures_dir=figures_dir,
        top_sources=args.top_sources,
        min_source_examples=args.min_source_examples,
        include_direct_class=args.include_direct_class,
    )

    write_summary_markdown(
        runs_df=runs_df,
        class_df=class_df,
        direct_df=direct_df,
        summary_path=output_dir / "analysis_summary.md",
    )

    print(json.dumps({"runs_analyzed": int(len(runs_df)), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
