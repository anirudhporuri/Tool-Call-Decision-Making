#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from mizani.formatters import percent_format

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from plotnine import (  # noqa: E402
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_col,
    geom_text,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_bw,
)


FAMILY_ORDER = {"llama": 0, "gemma": 1}
RUN_GROUP_ORDER = {"Prompting": 0, "Post-Training": 1, "CAI": 2, "Probe": 3, "Unknown": 99}
VARIANT_ORDER = {
    "Zero-shot": 0,
    "4-shot": 1,
    "SFT": 2,
    "DPO": 3,
    "DPO (Base)": 4,
    "DPO (From SFT)": 5,
    "Middle layer": 6,
    "75% depth layer": 7,
    "Last layer": 8,
}
GROUP_VARIANT_ORDER = {
    "Prompting": {"Zero-shot": 0, "4-shot": 1},
    "Post-Training": {"SFT": 0, "DPO": 1},
    "CAI": {"SFT": 0, "DPO (Base)": 1, "DPO (From SFT)": 2, "DPO": 3},
}
PROBE_LAYER_DISPLAY_BY_TAG = {
    "middle": "Middle layer",
    "layer_75pct": "75% depth layer",
    "75pct": "75% depth layer",
    "last": "Last layer",
}
PROBE_SOURCE_ORDER = {
    "Prompting Zero-shot": 0,
    "Prompting 4-shot": 1,
    "Post-Training SFT": 2,
    "Post-Training DPO": 3,
    "Unknown source": 99,
}
PROBE_SETTING_ORDER = {"n/a": -1, "Zero-shot": 0, "4-shot": 1, "Unknown": 2}
DEFAULT_LABEL_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
PRIMARY_BEHAVIOR_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
CLASS_DISPLAY = {
    "direct": "Direct",
    "tool_call": "Tool call",
    "request_for_info": "Request for info",
    "cannot_answer": "Cannot answer",
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
    "pending": "#adb5bd",
}
PREDICTION_COLORS = {
    "Tool call": "#1f78b4",
    "Request for info": "#2ca02c",
    "Cannot answer": "#d62728",
    "Direct": "#9467bd",
}
CORRUPTED_PROBE_SOURCE_VARIANTS: set[str] = set()
PROBE_PENDING_REASON = "Pending probe run"
PROBE_LAYER_COLORS = {
    "Middle layer": "#1f78b4",
    "75% depth layer": "#2ca02c",
    "Last layer": "#d62728",
}

RUN_COLUMNS = [
    "run_key",
    "run_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "probe_eval_setting",
    "raw_accuracy",
    "norm_accuracy",
    "raw_macro_f1",
    "norm_macro_f1",
    "n_examples",
]
CLASS_COLUMNS = [
    "run_key",
    "run_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "scoring",
    "behavior_class",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "support",
]
DIRECT_COLUMNS = [
    "run_key",
    "run_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "scoring",
    "n_examples",
    "direct_predictions",
    "direct_prediction_rate",
]
OUTCOME_COLUMNS = [
    "run_key",
    "run_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "outcome",
    "count",
    "fraction",
]
PRED_MIX_COLUMNS = [
    "run_key",
    "run_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "scoring",
    "label",
    "fraction",
]
PROBE_LAYER_COLUMNS = [
    "run_key",
    "run_display",
    "probe_parent_key",
    "probe_parent_display",
    "model_family",
    "run_group",
    "variant_label",
    "probe_source_variant",
    "is_probe",
    "is_placeholder",
    "probe_eval_setting",
    "raw_accuracy",
    "norm_accuracy",
    "raw_macro_f1",
    "norm_macro_f1",
    "raw_macro_precision",
    "norm_macro_precision",
    "raw_macro_recall",
    "norm_macro_recall",
    "agreement_with_model_pred_norm",
    "agreement_with_model_pred_raw",
    "agreement_with_model_pred_norm_restricted",
    "agreement_with_model_pred_raw_restricted",
    "n_examples",
    "used_in_main_plots",
    "accuracy_rank_within_probe",
    "macro_f1_rank_within_probe",
    "selection_reason",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_runs_dir = repo_root / "hmm"
    default_probe_runs_dir = repo_root / "hmm" / "Probe Evals"
    default_output_dir = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(
        description="Bar-only analysis for hmm snapshot (Prompting + Post-Training + CAI + Probe) with pending Gemma CAI placeholders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        default=str(default_runs_dir),
        help="Root containing hmm eval subfolders or a direct eval runs folder.",
    )
    parser.add_argument(
        "--probe-runs-dir",
        default=str(default_probe_runs_dir),
        help="Probe runs directory or a root containing a `Probe Evals` subfolder.",
    )
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Disable inferred placeholder rows for missing Gemma CAI eval runs.",
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


def infer_run_group_from_source_name(name: str) -> str:
    key = str(name).lower()
    if "prompting" in key:
        return "Prompting"
    if "post-training" in key or "post_training" in key:
        return "Post-Training"
    if "cai" in key:
        return "CAI"
    if "probe" in key:
        return "Probe"
    return "Unknown"


def _discover_leaf_run_dirs(parent: Path, required: Sequence[str]) -> List[Path]:
    if not parent.exists() or not parent.is_dir():
        return []
    found: List[Path] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        child_files = {p.name for p in child.iterdir() if p.is_file()}
        if set(required).issubset(child_files):
            found.append(child)
    return sorted(found)


def discover_run_dirs(runs_dir: Path) -> List[Tuple[Path, str]]:
    required = {"summary.json", "run_config.json"}
    if not runs_dir.exists():
        return []
    grouped: List[Tuple[Path, str]] = []
    subgroup_map = {
        "Prompting": runs_dir / "Prompting Evals",
        "Post-Training": runs_dir / "Post-Training Evals",
        "CAI": runs_dir / "CAI Evals",
    }
    has_subgroup_layout = any(path.exists() for path in subgroup_map.values())
    if has_subgroup_layout:
        for run_group, subgroup in subgroup_map.items():
            for run_dir in _discover_leaf_run_dirs(subgroup, required):
                grouped.append((run_dir, run_group))
        return sorted(grouped, key=lambda item: (item[1], item[0].name))

    inferred_group = infer_run_group_from_source_name(runs_dir.name)
    direct = _discover_leaf_run_dirs(runs_dir, required)
    return [(path, inferred_group) for path in direct]


def discover_probe_dirs(probe_runs_dir: Path) -> List[Path]:
    required = {"probe_evaluation_summary.json", "run_config.json"}
    if not probe_runs_dir.exists():
        return []
    if (probe_runs_dir / "Probe Evals").exists():
        return _discover_leaf_run_dirs(probe_runs_dir / "Probe Evals", required)
    return _discover_leaf_run_dirs(probe_runs_dir, required)


def infer_model_family(run_key: str, run_config: Dict[str, Any]) -> str:
    family = str(run_config.get("model_family", "")).lower().strip()
    model_name = str(run_config.get("model_name_or_path", "")).lower()
    haystack = f"{run_key.lower()} {family} {model_name}"
    if "llama" in haystack:
        return "llama"
    if "gemma" in haystack:
        return "gemma"
    return family or "unknown"


def infer_run_group(run_key: str, run_config: Dict[str, Any], source_group: str = "Unknown") -> str:
    if source_group in {"Prompting", "Post-Training", "CAI", "Probe"}:
        return source_group
    haystack = " ".join(
        [
            run_key.lower(),
            str(run_config.get("output_dir", "")).lower(),
            str(run_config.get("model_name_or_path", "")).lower(),
        ]
    )
    if "self_full" in haystack or "_model_eval" in haystack:
        return "CAI"
    if "sft_zeroshot_eval" in haystack or "dpo_zeroshot_eval" in haystack:
        return "Post-Training"
    if "probe" in haystack:
        return "Probe"
    if "4shot" in haystack or "4-shot" in haystack or "zeroshot" in haystack or "zero-shot" in haystack:
        return "Prompting"
    return "Unknown"


def infer_eval_variant_label(run_key: str, run_config: Dict[str, Any], run_group: str) -> str:
    haystack = " ".join(
        [
            run_key.lower(),
            str(run_config.get("output_dir", "")).lower(),
            str(run_config.get("model_name_or_path", "")).lower(),
        ]
    )

    if run_group == "CAI":
        if "dpo_from_sft" in haystack:
            return "DPO (From SFT)"
        if "dpo_base" in haystack:
            return "DPO (Base)"
        if "dpo" in haystack:
            return "DPO"
        if "sft" in haystack:
            return "SFT"
        return "Unknown"

    if run_group == "Post-Training":
        if "dpo" in haystack:
            return "DPO"
        if "sft" in haystack:
            return "SFT"
        return "Unknown"

    num_shots = int(run_config.get("num_shots", 0) or 0)
    if run_group == "Prompting":
        if "4shot" in haystack or "4-shot" in haystack or num_shots == 4:
            return "4-shot"
        if "zeroshot" in haystack or "zero-shot" in haystack or num_shots == 0:
            return "Zero-shot"
        return f"{num_shots}-shot"

    num_shots = int(run_config.get("num_shots", 0) or 0)
    if "dpo" in haystack:
        return "DPO"
    if "sft" in haystack:
        return "SFT"
    if "4shot" in haystack or "4-shot" in haystack or num_shots == 4:
        return "4-shot"
    if num_shots == 0:
        return "Zero-shot"
    return f"{num_shots}-shot"


def infer_probe_variant_label(layer_tag: str) -> str:
    key = str(layer_tag).strip().lower()
    if key in PROBE_LAYER_DISPLAY_BY_TAG:
        return PROBE_LAYER_DISPLAY_BY_TAG[key]
    if "75" in key:
        return "75% depth layer"
    if "mid" in key:
        return "Middle layer"
    if "last" in key:
        return "Last layer"
    return str(layer_tag)


def infer_probe_eval_setting(run_key: str, run_config: Dict[str, Any]) -> str:
    eval_path = str(run_config.get("eval_samples_jsonl", "")).lower()
    haystack = " ".join(
        [
            run_key.lower(),
            eval_path,
            str(run_config.get("output_dir", "")).lower(),
        ]
    )
    if "4shot" in haystack or "4-shot" in haystack:
        return "4-shot"
    if "zeroshot" in haystack or "zero-shot" in haystack or "probe_base" in haystack:
        return "Zero-shot"
    return "Unknown"


def infer_probe_source_variant(run_key: str, run_config: Dict[str, Any]) -> str:
    eval_path = str(run_config.get("eval_samples_jsonl", "")).lower()
    haystack = " ".join(
        [
            run_key.lower(),
            eval_path,
            str(run_config.get("output_dir", "")).lower(),
            str(run_config.get("model_name_or_path", "")).lower(),
        ]
    )
    num_shots = int(run_config.get("num_shots", 0) or 0)

    if "_sft_probe" in haystack or "sft_zeroshot_eval" in haystack:
        return "Post-Training SFT"
    if "_dpo_probe" in haystack or "dpo_zeroshot_eval" in haystack:
        return "Post-Training DPO"
    if "probe_4shot" in haystack or "4shot" in haystack or "4-shot" in haystack or num_shots == 4:
        return "Prompting 4-shot"
    if "probe_base" in haystack or "zeroshot" in haystack or "zero-shot" in haystack or num_shots == 0:
        return "Prompting Zero-shot"
    return "Unknown source"


def family_display_names(model_family: str) -> Tuple[str, str]:
    if model_family == "llama":
        return "Llama 3.2 3B", "Llama"
    if model_family == "gemma":
        return "Gemma 3 4B", "Gemma"
    return model_family, model_family


def make_run_display(family_short: str, run_group: str, variant_label: str) -> str:
    if run_group in {"Prompting", "Post-Training", "CAI"}:
        return f"{family_short} {run_group} {variant_label}"
    return f"{family_short} {variant_label}"


def short_family_display(model_family: str) -> str:
    if model_family == "llama":
        return "Llama"
    if model_family == "gemma":
        return "Gemma"
    return str(model_family).strip().title() or "Model"


def short_variant_display(run_group: str, variant_label: str) -> str:
    mapping: Dict[Tuple[str, str], str] = {
        ("Prompting", "Zero-shot"): "0-SHOT",
        ("Prompting", "4-shot"): "4-SHOT",
        ("Post-Training", "SFT"): "SFT",
        ("Post-Training", "DPO"): "DPO",
        ("CAI", "SFT"): "CAI-SFT",
        ("CAI", "DPO (Base)"): "CAI DPO BASE",
        ("CAI", "DPO (From SFT)"): "CAI DPO SFT",
        ("CAI", "DPO"): "CAI DPO",
    }
    return mapping.get((str(run_group), str(variant_label)), str(variant_label))


def short_run_display(model_family: str, run_group: str, variant_label: str) -> str:
    return f"{short_family_display(model_family)} {short_variant_display(run_group, variant_label)}"


def build_short_run_label_map(runs_subset_df: pd.DataFrame, include_family: bool = True) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    cols = ["run_display", "model_family", "run_group", "variant_label"]
    for _, row in runs_subset_df[cols].drop_duplicates().iterrows():
        run_display = str(row["run_display"])
        if include_family:
            label_map[run_display] = short_run_display(
                model_family=str(row["model_family"]),
                run_group=str(row["run_group"]),
                variant_label=str(row["variant_label"]),
            )
        else:
            label_map[run_display] = short_variant_display(
                run_group=str(row["run_group"]),
                variant_label=str(row["variant_label"]),
            )
    return label_map


def make_probe_run_display(
    family_short: str,
    probe_eval_setting: str,
    layer_label: str,
    probe_source_variant: str,
) -> str:
    if probe_source_variant and probe_source_variant != "Unknown source":
        return f"{family_short} Probe {probe_source_variant} ({layer_label})"
    if probe_eval_setting and probe_eval_setting != "Unknown":
        return f"{family_short} Probe Prompting {probe_eval_setting} ({layer_label})"
    return f"{family_short} Probe ({layer_label})"


def make_probe_parent_display(family_short: str, probe_eval_setting: str, probe_source_variant: str) -> str:
    if probe_source_variant and probe_source_variant != "Unknown source":
        return f"{family_short} Probe {probe_source_variant}"
    if probe_eval_setting and probe_eval_setting != "Unknown":
        return f"{family_short} Probe Prompting {probe_eval_setting}"
    return f"{family_short} Probe"


def build_probe_summary_like(layer_metrics: Dict[str, Any], n_examples: int) -> Dict[str, Any]:
    labels = list(layer_metrics.get("confusion_matrix_labels") or [])
    confusion_matrix = layer_metrics.get("confusion_matrix") or []
    class_report = layer_metrics.get("classification_report") or {}
    accuracy = float(layer_metrics.get("accuracy", class_report.get("accuracy", 0.0)) or 0.0)
    macro_f1 = float(
        layer_metrics.get(
            "macro_f1",
            (class_report.get("macro avg", {}) or {}).get("f1-score", 0.0),
        )
        or 0.0
    )
    scoring_payload = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": class_report,
        "confusion_matrix": confusion_matrix,
    }
    return {
        "label_order": labels,
        "raw": scoring_payload,
        "normalized": scoring_payload,
        "n_examples": int(n_examples),
    }


def probe_layer_score(layer_payload: Dict[str, Any]) -> Tuple[float, float]:
    probe_vs_gold = layer_payload.get("probe_vs_gold") or {}
    if not probe_vs_gold:
        return -1.0, -1.0
    accuracy = float(probe_vs_gold.get("accuracy", 0.0) or 0.0)
    macro_f1 = float(
        probe_vs_gold.get(
            "macro_f1",
            (probe_vs_gold.get("classification_report", {}) or {}).get("macro avg", {}).get("f1-score", 0.0),
        )
        or 0.0
    )
    return accuracy, macro_f1


def select_best_probe_layer(layers: Dict[str, Dict[str, Any]], layer_order: Sequence[str]) -> str | None:
    candidates = [tag for tag in layer_order if tag in layers]
    if not candidates:
        candidates = sorted(layers.keys())

    best_tag: str | None = None
    best_score = (-1.0, -1.0)
    for layer_tag in candidates:
        score = probe_layer_score(layers.get(layer_tag) or {})
        if score > best_score:
            best_tag = layer_tag
            best_score = score
    return best_tag


def extract_class_metrics(
    *,
    run_key: str,
    run_display: str,
    model_family: str,
    run_group: str,
    variant_label: str,
    probe_source_variant: str = "n/a",
    summary: Dict[str, Any],
    is_probe: bool = False,
    is_placeholder: bool = False,
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
                    "run_group": run_group,
                    "variant_label": variant_label,
                    "probe_source_variant": probe_source_variant,
                    "is_probe": bool(is_probe),
                    "is_placeholder": bool(is_placeholder),
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
    run_group: str,
    variant_label: str,
    probe_source_variant: str = "n/a",
    samples: List[Dict[str, Any]],
    label_order: Sequence[str],
    is_probe: bool = False,
    is_placeholder: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not samples:
        return [], [], []

    n_examples = len(samples)
    direct_rate_records: List[Dict[str, Any]] = []
    outcome_records: List[Dict[str, Any]] = []
    prediction_mix_records: List[Dict[str, Any]] = []

    for scoring, pred_col in (("raw", "pred_raw"), ("normalized", "pred_norm")):
        direct_predictions = sum(1 for row in samples if row.get(pred_col) == "direct")
        direct_rate_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": probe_source_variant,
                "is_probe": bool(is_probe),
                "is_placeholder": bool(is_placeholder),
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
                    "run_group": run_group,
                    "variant_label": variant_label,
                    "probe_source_variant": probe_source_variant,
                    "is_probe": bool(is_probe),
                    "is_placeholder": bool(is_placeholder),
                    "scoring": scoring,
                    "label": label,
                    "fraction": counts.get(label, 0) / max(n_examples, 1),
                }
            )

    outcome_counts = {name: 0 for name in OUTCOME_ORDER}
    for row in samples:
        gold = str(row.get("gold"))
        pred_raw = str(row.get("pred_raw"))
        pred_norm = str(row.get("pred_norm"))
        outcome_counts[normalization_outcome(gold, pred_raw, pred_norm)] += 1

    for outcome_name, count in outcome_counts.items():
        outcome_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": probe_source_variant,
                "is_probe": bool(is_probe),
                "is_placeholder": bool(is_placeholder),
                "outcome": outcome_name,
                "count": count,
                "fraction": count / max(n_examples, 1),
            }
        )

    return direct_rate_records, outcome_records, prediction_mix_records


def variant_rank_for_group(run_group: str, variant_label: str) -> int:
    group_order = GROUP_VARIANT_ORDER.get(run_group)
    if group_order is not None:
        return int(group_order.get(variant_label, 99))
    return int(VARIANT_ORDER.get(variant_label, 99))


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["family_rank"] = frame["model_family"].map(FAMILY_ORDER).fillna(99).astype(int)
    if "run_group" in frame.columns:
        frame["run_group_rank"] = frame["run_group"].map(RUN_GROUP_ORDER).fillna(99).astype(int)
    else:
        frame["run_group_rank"] = 99

    if {"run_group", "variant_label"}.issubset(frame.columns):
        frame["variant_rank"] = frame.apply(
            lambda row: variant_rank_for_group(str(row.get("run_group")), str(row.get("variant_label"))),
            axis=1,
        )
    else:
        frame["variant_rank"] = 99

    if "probe_source_variant" in frame.columns:
        frame["probe_source_rank"] = frame["probe_source_variant"].map(PROBE_SOURCE_ORDER).fillna(99).astype(int)
    else:
        frame["probe_source_rank"] = 99

    if "probe_eval_setting" in frame.columns:
        frame["probe_setting_rank"] = frame["probe_eval_setting"].map(PROBE_SETTING_ORDER).fillna(99).astype(int)
    else:
        frame["probe_setting_rank"] = 99

    return frame.sort_values(
        ["family_rank", "run_group_rank", "probe_source_rank", "probe_setting_rank", "variant_rank", "run_key"]
    ).reset_index(drop=True)


def enrich_probe_layer_selection(probe_layers_df: pd.DataFrame) -> pd.DataFrame:
    if probe_layers_df.empty:
        return probe_layers_df

    frame = probe_layers_df.copy()
    frame["accuracy_rank_within_probe"] = (
        frame.groupby("probe_parent_key")["norm_accuracy"].rank(method="dense", ascending=False).astype(int)
    )
    frame["macro_f1_rank_within_probe"] = (
        frame.groupby("probe_parent_key")["norm_macro_f1"].rank(method="dense", ascending=False).astype(int)
    )
    frame["selection_reason"] = ""

    eps = 1e-12
    for parent_key, group in frame.groupby("probe_parent_key", sort=False):
        if bool(group["is_placeholder"].all()):
            chosen_rows = frame[(frame["probe_parent_key"] == parent_key) & (frame["used_in_main_plots"])]
            if not chosen_rows.empty:
                frame.loc[chosen_rows.index, "selection_reason"] = PROBE_PENDING_REASON
            continue

        ranked = group.sort_values(["norm_accuracy", "norm_macro_f1"], ascending=False).reset_index(drop=True)
        if ranked.empty:
            continue
        best_idx = ranked.index[0]
        best_acc = float(ranked.loc[best_idx, "norm_accuracy"])
        best_f1 = float(ranked.loc[best_idx, "norm_macro_f1"])
        reason = "Highest accuracy"

        if len(ranked) > 1:
            second_acc = float(ranked.loc[1, "norm_accuracy"])
            second_f1 = float(ranked.loc[1, "norm_macro_f1"])
            if abs(best_acc - second_acc) <= eps:
                if best_f1 > second_f1 + eps:
                    reason = "Accuracy tie; higher macro-F1"
                else:
                    reason = "Tie after accuracy and macro-F1"

        chosen_rows = frame[(frame["probe_parent_key"] == parent_key) & (frame["used_in_main_plots"])]
        if not chosen_rows.empty:
            frame.loc[chosen_rows.index, "selection_reason"] = reason

    return frame


def build_probe_pending_rows(
    *,
    run_key: str,
    run_display: str,
    model_family: str,
    run_group: str,
    variant_label: str,
    probe_source_variant: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    class_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    pred_mix_rows: List[Dict[str, Any]] = []

    for scoring in ("raw", "normalized"):
        direct_rows.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": probe_source_variant,
                "is_probe": True,
                "is_placeholder": True,
                "scoring": scoring,
                "n_examples": 0,
                "direct_predictions": 0,
                "direct_prediction_rate": 0.0,
            }
        )
        for label in DEFAULT_LABEL_ORDER:
            pred_mix_rows.append(
                {
                    "run_key": run_key,
                    "run_display": run_display,
                    "model_family": model_family,
                    "run_group": run_group,
                    "variant_label": variant_label,
                    "probe_source_variant": probe_source_variant,
                    "is_probe": True,
                    "is_placeholder": True,
                    "scoring": scoring,
                    "label": label,
                    "fraction": 0.0,
                }
            )
        for behavior_class in PRIMARY_BEHAVIOR_CLASSES:
            class_rows.append(
                {
                    "run_key": run_key,
                    "run_display": run_display,
                    "model_family": model_family,
                    "run_group": run_group,
                    "variant_label": variant_label,
                    "probe_source_variant": probe_source_variant,
                    "is_probe": True,
                    "is_placeholder": True,
                    "scoring": scoring,
                    "behavior_class": behavior_class,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0.0,
                }
            )

    for outcome in OUTCOME_ORDER:
        outcome_rows.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": probe_source_variant,
                "is_probe": True,
                "is_placeholder": True,
                "outcome": outcome,
                "count": 0,
                "fraction": 0.0,
            }
        )

    return class_rows, direct_rows, outcome_rows, pred_mix_rows


def enforce_unique_run_displays(
    runs_df: pd.DataFrame,
    class_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    prediction_mix_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate_labels = set(runs_df["run_display"].value_counts()[lambda s: s > 1].index.tolist())
    if not duplicate_labels:
        return runs_df, class_df, direct_df, outcome_df, prediction_mix_df

    display_map: Dict[str, str] = {}
    for _, row in runs_df.iterrows():
        run_key = str(row["run_key"])
        run_display = str(row["run_display"])
        if run_display in duplicate_labels:
            display_map[run_key] = f"{run_display} ({run_key})"
        else:
            display_map[run_key] = run_display

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["run_display"] = out["run_key"].map(display_map).fillna(out["run_display"])
        return out

    return (
        _apply(runs_df),
        _apply(class_df),
        _apply(direct_df),
        _apply(outcome_df),
        _apply(prediction_mix_df),
    )


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


def save_plot_multi(plot_obj: Any, *, figures_dir: Path, base_name: str, width: float, height: float) -> None:
    save_plot(plot_obj, figures_dir / f"{base_name}.pdf", width=width, height=height)


def pending_annotation_df(runs_df: pd.DataFrame, run_order: List[str], y_value: float) -> pd.DataFrame:
    pending = runs_df[runs_df["is_placeholder"]][["run_display"]].drop_duplicates().copy()
    if pending.empty:
        return pending
    pending["run_display"] = pd.Categorical(pending["run_display"], categories=run_order, ordered=True)
    pending["y"] = y_value
    pending["label"] = "PENDING"
    return pending


def infer_missing_gemma_cai_eval_run_keys(existing_run_keys: Iterable[str]) -> List[str]:
    existing = set(existing_run_keys)
    expected = {
        "gemma_self_full_sft_model_eval",
        "gemma_self_full_dpo_base_model_eval",
        "gemma_self_full_dpo_from_sft_model_eval",
    }
    for run_key in existing:
        if run_key.startswith("llama_self_full_") and run_key.endswith("_eval"):
            expected.add("gemma_self_full_" + run_key[len("llama_self_full_") :])
    return sorted(key for key in expected if key not in existing)


def build_placeholder_rows(missing_run_keys: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    pred_mix_rows: List[Dict[str, Any]] = []

    _, family_short = family_display_names("gemma")

    for run_key in missing_run_keys:
        run_group = "CAI"
        variant = infer_eval_variant_label(
            run_key,
            {"output_dir": run_key, "model_name_or_path": run_key, "num_shots": 0},
            run_group=run_group,
        )
        display = make_run_display(family_short, run_group, variant)

        run_rows.append(
            {
                "run_key": run_key,
                "run_display": display,
                "model_family": "gemma",
                "run_group": run_group,
                "variant_label": variant,
                "probe_source_variant": "n/a",
                "is_probe": False,
                "is_placeholder": True,
                "probe_eval_setting": "n/a",
                "raw_accuracy": 0.0,
                "norm_accuracy": 0.0,
                "raw_macro_f1": 0.0,
                "norm_macro_f1": 0.0,
                "n_examples": 0,
            }
        )

        for scoring in ("raw", "normalized"):
            direct_rows.append(
                {
                    "run_key": run_key,
                    "run_display": display,
                    "model_family": "gemma",
                    "run_group": run_group,
                    "variant_label": variant,
                    "probe_source_variant": "n/a",
                    "is_probe": False,
                    "is_placeholder": True,
                    "scoring": scoring,
                    "n_examples": 0,
                    "direct_predictions": 0,
                    "direct_prediction_rate": 0.0,
                }
            )
            for label in DEFAULT_LABEL_ORDER:
                pred_mix_rows.append(
                    {
                        "run_key": run_key,
                        "run_display": display,
                        "model_family": "gemma",
                        "run_group": run_group,
                        "variant_label": variant,
                        "probe_source_variant": "n/a",
                        "is_probe": False,
                        "is_placeholder": True,
                        "scoring": scoring,
                        "label": label,
                        "fraction": 0.0,
                    }
                )

        for outcome in OUTCOME_ORDER:
            outcome_rows.append(
                {
                    "run_key": run_key,
                    "run_display": display,
                    "model_family": "gemma",
                    "run_group": run_group,
                    "variant_label": variant,
                    "probe_source_variant": "n/a",
                    "is_probe": False,
                    "is_placeholder": True,
                    "outcome": outcome,
                    "count": 0,
                    "fraction": 0.0,
                }
            )

        for scoring in ("raw", "normalized"):
            for behavior_class in PRIMARY_BEHAVIOR_CLASSES:
                class_rows.append(
                    {
                        "run_key": run_key,
                        "run_display": display,
                        "model_family": "gemma",
                        "run_group": run_group,
                        "variant_label": variant,
                        "probe_source_variant": "n/a",
                        "is_probe": False,
                        "is_placeholder": True,
                        "scoring": scoring,
                        "behavior_class": behavior_class,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "support": 0.0,
                    }
                )

    return run_rows, class_rows, direct_rows, outcome_rows, pred_mix_rows


def write_summary_markdown(*, runs_df: pd.DataFrame, missing_placeholder_keys: Sequence[str], summary_path: Path) -> None:
    lines = [
        "# Bar-Only Analysis Summary",
        "",
        f"- Generated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        f"- Runs analyzed: **{len(runs_df)}**",
        f"- Placeholder runs injected: **{len(missing_placeholder_keys)}**",
    ]
    if missing_placeholder_keys:
        lines.append("")
        lines.append("## Pending Gemma CAI Placeholders")
        lines.append("")
        for key in missing_placeholder_keys:
            lines.append(f"- `{key}`")
    lines.append("")
    lines.append("Heatmaps are intentionally omitted in this bar-only report.")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def fmt_dec(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_signed(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def fmt_latex_number(value: float, *, signed: bool = False, digits: int = 3, bold: bool = False) -> str:
    txt = f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def column_extrema_mask(
    numeric_rows: Sequence[Sequence[float]],
    *,
    preferences: Sequence[str],
    eligible_rows: Sequence[bool] | None = None,
    eps: float = 1e-12,
) -> List[List[bool]]:
    if not numeric_rows:
        return []
    n_cols = len(numeric_rows[0])
    mask = [[False for _ in range(n_cols)] for _ in range(len(numeric_rows))]
    if eligible_rows is None:
        eligible_rows = [True] * len(numeric_rows)
    for col_idx in range(n_cols):
        pref = preferences[col_idx] if col_idx < len(preferences) else "none"
        if pref not in {"max", "min"}:
            continue
        eligible_indices = [i for i, ok in enumerate(eligible_rows) if bool(ok)]
        if not eligible_indices:
            continue
        values = [float(numeric_rows[i][col_idx]) for i in eligible_indices]
        target = max(values) if pref == "max" else min(values)
        for row_idx in eligible_indices:
            v = float(numeric_rows[row_idx][col_idx])
            if abs(v - target) <= eps:
                mask[row_idx][col_idx] = True
    return mask


def build_latex_rows_with_bold(
    *,
    labels: Sequence[str],
    numeric_rows: Sequence[Sequence[float]],
    preferences: Sequence[str],
    eligible_rows: Sequence[bool] | None = None,
    signed_cols: Sequence[int] = (),
    digits: int = 3,
) -> List[List[str]]:
    extrema = column_extrema_mask(
        numeric_rows,
        preferences=preferences,
        eligible_rows=eligible_rows,
    )
    signed_col_set = set(signed_cols)
    out: List[List[str]] = []
    for row_idx, label in enumerate(labels):
        row_out = [latex_escape(label)]
        for col_idx, val in enumerate(numeric_rows[row_idx]):
            row_out.append(
                fmt_latex_number(
                    float(val),
                    signed=(col_idx in signed_col_set),
                    digits=digits,
                    bold=extrema[row_idx][col_idx],
                )
            )
        out.append(row_out)
    return out


def write_latex_table_text(
    *,
    output_path: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str,
    caption: str,
    label: str,
) -> None:
    lines: List[str] = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join(headers) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_ordered_subset(df: pd.DataFrame, run_order: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["run_display"] = pd.Categorical(out["run_display"], categories=list(run_order), ordered=True)
    return out.sort_values(["run_display"]).reset_index(drop=True)


def write_chart_tables(
    *,
    runs_df: pd.DataFrame,
    class_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    prediction_mix_df: pd.DataFrame,
    probe_layers_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    run_order = runs_df["run_display"].tolist()
    no_probe_runs = runs_df[~runs_df["is_probe"].fillna(False)].copy()
    no_probe_order = no_probe_runs["run_display"].tolist()
    run_is_eligible_all = {
        str(row["run_display"]): (not bool(row["is_placeholder"]))
        for _, row in runs_df[["run_display", "is_placeholder"]].drop_duplicates().iterrows()
    }
    run_is_eligible_no_probe = {
        str(row["run_display"]): (not bool(row["is_placeholder"]))
        for _, row in no_probe_runs[["run_display", "is_placeholder"]].drop_duplicates().iterrows()
    }

    def _write_run_metric_table(
        *,
        base_name: str,
        metric_col: str,
        metric_header: str,
        run_subset: pd.DataFrame,
        caption: str,
        label: str,
    ) -> None:
        ordered = run_ordered_subset(run_subset, run_subset["run_display"].tolist())
        labels = [str(r["run_display"]) for _, r in ordered.iterrows()]
        numeric_rows = [[float(r[metric_col])] for _, r in ordered.iterrows()]
        eligible_rows = [not bool(r.get("is_placeholder", False)) for _, r in ordered.iterrows()]
        rows = build_latex_rows_with_bold(
            labels=labels,
            numeric_rows=numeric_rows,
            preferences=["max"],
            eligible_rows=eligible_rows,
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_table_latex.txt",
            headers=["Run", metric_header],
            rows=rows,
            align="lr",
            caption=caption,
            label=label,
        )

    _write_run_metric_table(
        base_name="normalized_accuracy_by_run",
        metric_col="norm_accuracy",
        metric_header="Accuracy",
        run_subset=run_ordered_subset(runs_df, run_order),
        caption="Accuracy by run (all runs).",
        label="tab:normalized-accuracy-by-run",
    )
    _write_run_metric_table(
        base_name="normalized_macro_f1_by_run",
        metric_col="norm_macro_f1",
        metric_header="Macro-F1",
        run_subset=run_ordered_subset(runs_df, run_order),
        caption="Macro-F1 by run (all runs).",
        label="tab:normalized-macrof1-by-run",
    )
    _write_run_metric_table(
        base_name="normalized_accuracy_by_run_without_probe",
        metric_col="norm_accuracy",
        metric_header="Accuracy",
        run_subset=run_ordered_subset(no_probe_runs, no_probe_order),
        caption="Accuracy by run (no probe).",
        label="tab:normalized-accuracy-by-run-no-probe",
    )
    _write_run_metric_table(
        base_name="normalized_macro_f1_by_run_without_probe",
        metric_col="norm_macro_f1",
        metric_header="Macro-F1",
        run_subset=run_ordered_subset(no_probe_runs, no_probe_order),
        caption="Macro-F1 by run (no probe).",
        label="tab:normalized-macrof1-by-run-no-probe",
    )

    class_norm = class_df[
        (class_df["scoring"] == "normalized") & (class_df["behavior_class"].isin(PRIMARY_BEHAVIOR_CLASSES))
    ].copy()
    class_metric_specs = [
        ("per_class_accuracy_bar_by_run", "accuracy", "Per-class one-vs-rest accuracy (all runs)."),
        ("per_class_precision_bar_by_run", "precision", "Per-class precision (all runs)."),
        ("per_class_recall_bar_by_run", "recall", "Per-class recall (all runs)."),
        ("per_class_f1_bar_by_run", "f1", "Per-class F1 (all runs)."),
    ]
    class_cols = ["tool_call", "request_for_info", "cannot_answer"]
    class_headers = [CLASS_DISPLAY[c] for c in class_cols]

    for base_name, metric_col, caption in class_metric_specs:
        sub = run_ordered_subset(class_norm, run_order)
        pivot = (
            sub.pivot_table(index="run_display", columns="behavior_class", values=metric_col, aggfunc="first")
            .reindex(columns=class_cols)
            .reset_index()
        )
        labels = [str(row["run_display"]) for _, row in pivot.iterrows()]
        numeric_rows = [
            [
                float(row.get("tool_call", 0.0)),
                float(row.get("request_for_info", 0.0)),
                float(row.get("cannot_answer", 0.0)),
            ]
            for _, row in pivot.iterrows()
        ]
        rows = build_latex_rows_with_bold(
            labels=labels,
            numeric_rows=numeric_rows,
            preferences=["max", "max", "max"],
            eligible_rows=[run_is_eligible_all.get(label, True) for label in labels],
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_table_latex.txt",
            headers=["Run"] + class_headers,
            rows=rows,
            align="lrrr",
            caption=caption,
            label=f"tab:{base_name}",
        )

        no_probe_sub = sub[~sub["is_probe"].fillna(False)].copy()
        no_probe_sub = run_ordered_subset(no_probe_sub, no_probe_order)
        pivot_no_probe = (
            no_probe_sub.pivot_table(index="run_display", columns="behavior_class", values=metric_col, aggfunc="first")
            .reindex(columns=class_cols)
            .reset_index()
        )
        labels_no_probe = [str(row["run_display"]) for _, row in pivot_no_probe.iterrows()]
        numeric_rows_no_probe = [
            [
                float(row.get("tool_call", 0.0)),
                float(row.get("request_for_info", 0.0)),
                float(row.get("cannot_answer", 0.0)),
            ]
            for _, row in pivot_no_probe.iterrows()
        ]
        rows_no_probe = build_latex_rows_with_bold(
            labels=labels_no_probe,
            numeric_rows=numeric_rows_no_probe,
            preferences=["max", "max", "max"],
            eligible_rows=[run_is_eligible_no_probe.get(label, True) for label in labels_no_probe],
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_without_probe_table_latex.txt",
            headers=["Run"] + class_headers,
            rows=rows_no_probe,
            align="lrrr",
            caption=caption.replace("(all runs)", "(no probe)"),
            label=f"tab:{base_name}-no-probe",
        )

    def _write_direct_table(base_name: str, subset: pd.DataFrame, caption: str, label: str) -> None:
        pivot = (
            subset.pivot_table(index="run_display", columns="scoring", values="direct_prediction_rate", aggfunc="first")
            .reset_index()
        )
        for col in [c for c in pivot.columns if c != "run_display"]:
            pivot[col] = pivot[col].fillna(0.0)
        labels = [str(row["run_display"]) for _, row in pivot.iterrows()]
        numeric_rows = []
        for _, row in pivot.iterrows():
            raw = float(row.get("raw", 0.0))
            norm = float(row.get("normalized", 0.0))
            numeric_rows.append([raw, norm, norm - raw])
        rows = build_latex_rows_with_bold(
            labels=labels,
            numeric_rows=numeric_rows,
            preferences=["min", "min", "min"],
            eligible_rows=[run_is_eligible_all.get(label, True) for label in labels],
            signed_cols=[2],
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_table_latex.txt",
            headers=["Run", "Raw Direct Rate", "Norm Direct Rate", "$\\Delta$ Direct Rate"],
            rows=rows,
            align="lrrr",
            caption=caption,
            label=label,
        )

    direct_all = run_ordered_subset(direct_df, run_order)
    _write_direct_table(
        base_name="unsupported_direct_prediction_rate_by_run",
        subset=direct_all,
        caption="Unsupported direct prediction rate by run (all runs).",
        label="tab:unsupported-direct-rate-by-run",
    )
    direct_no_probe = direct_all[~direct_all["is_probe"].fillna(False)].copy()
    direct_no_probe = run_ordered_subset(direct_no_probe, no_probe_order)
    _write_direct_table(
        base_name="unsupported_direct_prediction_rate_by_run_without_probe",
        subset=direct_no_probe,
        caption="Unsupported direct prediction rate by run (no probe).",
        label="tab:unsupported-direct-rate-by-run-no-probe",
    )

    def _write_outcome_table(base_name: str, subset: pd.DataFrame, caption: str, label: str) -> None:
        pivot = (
            subset.pivot_table(index="run_display", columns="outcome", values="fraction", aggfunc="first")
            .reindex(columns=OUTCOME_ORDER)
            .reset_index()
        )
        for col in [c for c in pivot.columns if c != "run_display"]:
            pivot[col] = pivot[col].fillna(0.0)
        labels = [str(row["run_display"]) for _, row in pivot.iterrows()]
        numeric_rows = [
            [
                float(row.get("stay_correct", 0.0)),
                float(row.get("fixed", 0.0)),
                float(row.get("broken", 0.0)),
                float(row.get("stay_wrong", 0.0)),
            ]
            for _, row in pivot.iterrows()
        ]
        rows = build_latex_rows_with_bold(
            labels=labels,
            numeric_rows=numeric_rows,
            preferences=["max", "max", "min", "min"],
            eligible_rows=[run_is_eligible_all.get(label, True) for label in labels],
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_table_latex.txt",
            headers=["Run", "Stay Correct", "Fixed", "Broken", "Stay Wrong"],
            rows=rows,
            align="lrrrr",
            caption=caption,
            label=label,
        )

    outcome_all = run_ordered_subset(outcome_df, run_order)
    _write_outcome_table(
        base_name="normalization_outcome_breakdown_by_run",
        subset=outcome_all,
        caption="Normalization outcome fractions by run (all runs).",
        label="tab:normalization-outcome-by-run",
    )
    outcome_no_probe = outcome_all[~outcome_all["is_probe"].fillna(False)].copy()
    outcome_no_probe = run_ordered_subset(outcome_no_probe, no_probe_order)
    _write_outcome_table(
        base_name="normalization_outcome_breakdown_by_run_without_probe",
        subset=outcome_no_probe,
        caption="Normalization outcome fractions by run (no probe).",
        label="tab:normalization-outcome-by-run-no-probe",
    )

    def _write_prediction_mix_table(base_name: str, subset: pd.DataFrame, caption: str, label: str) -> None:
        normalized = subset[subset["scoring"] == "normalized"].copy()
        pivot = (
            normalized.pivot_table(index="run_display", columns="label", values="fraction", aggfunc="first")
            .reindex(columns=DEFAULT_LABEL_ORDER)
            .reset_index()
        )
        for col in [c for c in pivot.columns if c != "run_display"]:
            pivot[col] = pivot[col].fillna(0.0)
        labels = [str(row["run_display"]) for _, row in pivot.iterrows()]
        numeric_rows = [
            [
                float(row.get("direct", 0.0)),
                float(row.get("tool_call", 0.0)),
                float(row.get("request_for_info", 0.0)),
                float(row.get("cannot_answer", 0.0)),
            ]
            for _, row in pivot.iterrows()
        ]
        rows = build_latex_rows_with_bold(
            labels=labels,
            numeric_rows=numeric_rows,
            preferences=["min", "max", "max", "max"],
            eligible_rows=[run_is_eligible_all.get(label, True) for label in labels],
        )
        write_latex_table_text(
            output_path=tables_dir / f"{base_name}_table_latex.txt",
            headers=["Run", "Direct", "Tool Call", "Request for Info", "Cannot Answer"],
            rows=rows,
            align="lrrrr",
            caption=caption,
            label=label,
        )

    pred_all = run_ordered_subset(prediction_mix_df, run_order)
    _write_prediction_mix_table(
        base_name="normalized_prediction_mix_by_run",
        subset=pred_all,
        caption="Prediction mix by run (all runs, normalized scoring).",
        label="tab:prediction-mix-by-run",
    )
    pred_no_probe = pred_all[~pred_all["is_probe"].fillna(False)].copy()
    pred_no_probe = run_ordered_subset(pred_no_probe, no_probe_order)
    _write_prediction_mix_table(
        base_name="normalized_prediction_mix_by_run_without_probe",
        subset=pred_no_probe,
        caption="Prediction mix by run (no probe, normalized scoring).",
        label="tab:prediction-mix-by-run-no-probe",
    )

    if not probe_layers_df.empty:
        ordered_probe_layers = sort_runs(probe_layers_df)
        layer_order = ["Middle layer", "75% depth layer", "Last layer"]
        parent_pending_map = (
            ordered_probe_layers.groupby("probe_parent_display", observed=False)["is_placeholder"].all().to_dict()
        )

        def _write_probe_layer_table(base_name: str, metric_col: str, caption: str, label: str) -> None:
            pivot = (
                ordered_probe_layers.pivot_table(
                    index="probe_parent_display",
                    columns="variant_label",
                    values=metric_col,
                    aggfunc="first",
                )
                .reindex(columns=layer_order)
                .reset_index()
            )
            for col in [c for c in pivot.columns if c != "probe_parent_display"]:
                pivot[col] = pivot[col].fillna(0.0)
            best = ordered_probe_layers[ordered_probe_layers["used_in_main_plots"].fillna(False)][
                ["probe_parent_display", "variant_label", "selection_reason"]
            ].drop_duplicates()
            merged = pivot.merge(best, on="probe_parent_display", how="left")
            rows = []
            for _, row in merged.iterrows():
                layer_values = [
                    float(row.get("Middle layer", 0.0)),
                    float(row.get("75% depth layer", 0.0)),
                    float(row.get("Last layer", 0.0)),
                ]
                parent_display = str(row["probe_parent_display"])
                is_pending_parent = bool(parent_pending_map.get(parent_display, False))
                if is_pending_parent:
                    layer_cells = [
                        fmt_latex_number(layer_values[0], bold=False),
                        fmt_latex_number(layer_values[1], bold=False),
                        fmt_latex_number(layer_values[2], bold=False),
                    ]
                    chosen_layer = "PENDING"
                    selection_reason = PROBE_PENDING_REASON
                else:
                    row_best = max(layer_values)
                    layer_cells = [
                        fmt_latex_number(layer_values[0], bold=abs(layer_values[0] - row_best) <= 1e-12),
                        fmt_latex_number(layer_values[1], bold=abs(layer_values[1] - row_best) <= 1e-12),
                        fmt_latex_number(layer_values[2], bold=abs(layer_values[2] - row_best) <= 1e-12),
                    ]
                    chosen_layer = str(row.get("variant_label", ""))
                    selection_reason = str(row.get("selection_reason", ""))
                rows.append(
                    [
                        latex_escape(parent_display),
                        layer_cells[0],
                        layer_cells[1],
                        layer_cells[2],
                        latex_escape(chosen_layer),
                        latex_escape(selection_reason),
                    ]
                )
            write_latex_table_text(
                output_path=tables_dir / f"{base_name}_table_latex.txt",
                headers=["Probe Setup", "Middle", "75% Depth", "Last", "Chosen Layer", "Selection Reason"],
                rows=rows,
                align="lrrrll",
                caption=caption,
                label=label,
            )

        _write_probe_layer_table(
            base_name="probe_layer_accuracy_comparison",
            metric_col="norm_accuracy",
            caption="Probe-layer accuracy comparison for each probe setup.",
            label="tab:probe-layer-accuracy-comparison",
        )
        _write_probe_layer_table(
            base_name="probe_layer_macro_f1_comparison",
            metric_col="norm_macro_f1",
            caption="Probe-layer macro-F1 comparison for each probe setup.",
            label="tab:probe-layer-macrof1-comparison",
        )
        _write_probe_layer_table(
            base_name="probe_layer_macro_precision_comparison",
            metric_col="norm_macro_precision",
            caption="Probe-layer macro precision comparison for each probe setup.",
            label="tab:probe-layer-macroprecision-comparison",
        )
        _write_probe_layer_table(
            base_name="probe_layer_macro_recall_comparison",
            metric_col="norm_macro_recall",
            caption="Probe-layer macro recall comparison for each probe setup.",
            label="tab:probe-layer-macrorecall-comparison",
        )
        _write_probe_layer_table(
            base_name="probe_layer_model_agreement_norm_comparison",
            metric_col="agreement_with_model_pred_norm",
            caption="Probe-layer agreement with model normalized predictions for each probe setup.",
            label="tab:probe-layer-model-agreement-norm-comparison",
        )
        _write_probe_layer_table(
            base_name="probe_layer_model_agreement_raw_comparison",
            metric_col="agreement_with_model_pred_raw",
            caption="Probe-layer agreement with model raw predictions for each probe setup.",
            label="tab:probe-layer-model-agreement-raw-comparison",
        )

    non_probe_runs = runs_df[~runs_df["is_probe"].fillna(False)].copy()
    norm_delta = non_probe_runs.copy()
    norm_delta["delta_accuracy"] = norm_delta["norm_accuracy"] - norm_delta["raw_accuracy"]
    norm_delta["delta_macro_f1"] = norm_delta["norm_macro_f1"] - norm_delta["raw_macro_f1"]
    direct_pivot = (
        direct_df[~direct_df["is_probe"].fillna(False)]
        .pivot_table(index="run_display", columns="scoring", values="direct_prediction_rate", aggfunc="first")
        .reset_index()
    )
    for col in [c for c in direct_pivot.columns if c != "run_display"]:
        direct_pivot[col] = direct_pivot[col].fillna(0.0)
    norm_delta = norm_delta.merge(direct_pivot, on="run_display", how="left")
    for col in ("raw", "normalized"):
        if col not in norm_delta.columns:
            norm_delta[col] = 0.0
    norm_delta[["raw", "normalized"]] = norm_delta[["raw", "normalized"]].fillna(0.0)
    norm_delta["delta_direct_rate"] = norm_delta["normalized"] - norm_delta["raw"]
    norm_delta = run_ordered_subset(norm_delta, no_probe_order)

    norm_labels = [str(row["run_display"]) for _, row in norm_delta.iterrows()]
    norm_numeric_rows = [
        [
            float(row["raw_accuracy"]),
            float(row["norm_accuracy"]),
            float(row["delta_accuracy"]),
            float(row["raw_macro_f1"]),
            float(row["norm_macro_f1"]),
            float(row["delta_macro_f1"]),
            float(row["raw"]),
            float(row["normalized"]),
            float(row["delta_direct_rate"]),
        ]
        for _, row in norm_delta.iterrows()
    ]
    raw_norm_rows = build_latex_rows_with_bold(
        labels=norm_labels,
        numeric_rows=norm_numeric_rows,
        preferences=["max", "max", "max", "max", "max", "max", "min", "min", "min"],
        eligible_rows=[not bool(row.get("is_placeholder", False)) for _, row in norm_delta.iterrows()],
        signed_cols=[2, 5, 8],
    )
    write_latex_table_text(
        output_path=tables_dir / "byte_normalization_effect_table_latex.txt",
        headers=[
            "Run",
            "Raw Acc",
            "Norm Acc",
            "$\\Delta$ Acc",
            "Raw Macro-F1",
            "Norm Macro-F1",
            "$\\Delta$ Macro-F1",
            "Raw Direct",
            "Norm Direct",
            "$\\Delta$ Direct",
        ],
        rows=raw_norm_rows,
        align="lrrrrrrrrr",
        caption="Byte-normalization impact by run (no probe).",
        label="tab:byte-normalization-impact",
    )
    write_latex_table_text(
        output_path=tables_dir / "raw_vs_normalized_change_table_latex.txt",
        headers=[
            "Run",
            "Raw Acc",
            "Norm Acc",
            "$\\Delta$ Acc",
            "Raw Macro-F1",
            "Norm Macro-F1",
            "$\\Delta$ Macro-F1",
        ],
        rows=[
            [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
            for r in raw_norm_rows
        ],
        align="lrrrrrr",
        caption="Raw vs. normalized scoring comparison by run (no probe).",
        label="tab:raw-vs-normalized-scoring",
    )

def make_bar_plots(
    *,
    runs_df: pd.DataFrame,
    class_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    prediction_mix_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    run_order = runs_df["run_display"].tolist()
    runs_plot = apply_run_order(runs_df, run_order)
    all_runs_suffix = "(All Runs)"
    no_probe_suffix = "(No Probe)"

    runs_plot["fill_group"] = runs_plot.apply(
        lambda row: "pending" if bool(row["is_placeholder"]) else str(row["model_family"]),
        axis=1,
    )

    runs_plot["norm_accuracy_label"] = runs_plot.apply(
        lambda row: "PENDING" if bool(row["is_placeholder"]) else f"{float(row['norm_accuracy']):.1%}",
        axis=1,
    )
    runs_plot["norm_macro_f1_label"] = runs_plot.apply(
        lambda row: "PENDING" if bool(row["is_placeholder"]) else f"{float(row['norm_macro_f1']):.1%}",
        axis=1,
    )
    runs_plot["norm_accuracy_label_pos"] = runs_plot["norm_accuracy"] + 0.03
    runs_plot["norm_macro_f1_label_pos"] = runs_plot["norm_macro_f1"] + 0.03
    no_probe_runs = runs_plot[~runs_plot["is_probe"].fillna(False)].copy()
    no_probe_order = no_probe_runs["run_display"].tolist()

    accuracy_plot = (
        ggplot(runs_plot, aes(x="run_display", y="norm_accuracy", fill="fill_group"))
        + geom_col(width=0.72)
        + geom_text(aes(y="norm_accuracy_label_pos", label="norm_accuracy_label"), size=10, ha="left")
        + coord_flip()
        + scale_fill_manual(values=FAMILY_COLORS)
        + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.08), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + labs(title=f"Accuracy on When2Call {all_runs_suffix}", x="", y="Accuracy", fill="Model Family")
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(accuracy_plot, figures_dir=figures_dir, base_name="normalized_accuracy_by_run", width=11, height=6)

    macrof1_plot = (
        ggplot(runs_plot, aes(x="run_display", y="norm_macro_f1", fill="fill_group"))
        + geom_col(width=0.72)
        + geom_text(aes(y="norm_macro_f1_label_pos", label="norm_macro_f1_label"), size=10, ha="left")
        + coord_flip()
        + scale_fill_manual(values=FAMILY_COLORS)
        + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.08), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + labs(title=f"Macro-F1 on When2Call {all_runs_suffix}", x="", y="Macro-F1", fill="Model Family")
        + theme_bw()
        + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
    )
    save_plot_multi(macrof1_plot, figures_dir=figures_dir, base_name="normalized_macro_f1_by_run", width=11, height=6)

    if not no_probe_runs.empty:
        no_probe_runs["norm_accuracy_label_pos"] = no_probe_runs["norm_accuracy"] + 0.03
        no_probe_runs["norm_macro_f1_label_pos"] = no_probe_runs["norm_macro_f1"] + 0.03

        no_probe_acc_plot = (
            ggplot(no_probe_runs, aes(x="run_display", y="norm_accuracy", fill="fill_group"))
            + geom_col(width=0.72)
            + geom_text(aes(y="norm_accuracy_label_pos", label="norm_accuracy_label"), size=10, ha="left")
            + coord_flip()
            + scale_fill_manual(values=FAMILY_COLORS)
            + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.08), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            + labs(
                title=f"Accuracy on When2Call {no_probe_suffix}",
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
            ggplot(no_probe_runs, aes(x="run_display", y="norm_macro_f1", fill="fill_group"))
            + geom_col(width=0.72)
            + geom_text(aes(y="norm_macro_f1_label_pos", label="norm_macro_f1_label"), size=10, ha="left")
            + coord_flip()
            + scale_fill_manual(values=FAMILY_COLORS)
            + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.08), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            + labs(
                title=f"Macro-F1 on When2Call {no_probe_suffix}",
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

    class_plot_df = class_df[
        (class_df["scoring"] == "normalized") & (class_df["behavior_class"].isin(PRIMARY_BEHAVIOR_CLASSES))
    ].copy()
    if not class_plot_df.empty:
        class_plot_df = apply_run_order(class_plot_df, run_order)
        class_plot_df["behavior_class_display"] = class_plot_df["behavior_class"].map(CLASS_DISPLAY)
        class_pending = pending_annotation_df(runs_plot, run_order, y_value=0.03)
        per_class_metrics = [
            ("accuracy", "Accuracy (One-vs-Rest)", "per_class_accuracy_bar_by_run", "Per-Class Accuracy"),
            ("precision", "Precision", "per_class_precision_bar_by_run", "Per-Class Precision"),
            ("recall", "Recall", "per_class_recall_bar_by_run", "Per-Class Recall"),
            ("f1", "F1", "per_class_f1_bar_by_run", "Per-Class F1"),
        ]

        for metric_col, y_label, base_name, title_suffix in per_class_metrics:
            metric_plot = (
                ggplot(class_plot_df, aes(x="run_display", y=metric_col, fill="behavior_class_display"))
                + geom_col(position=position_dodge(width=0.78), width=0.7)
                + coord_flip()
                + scale_fill_manual(values=PREDICTION_COLORS)
                + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                + labs(title=f"{title_suffix} on When2Call {all_runs_suffix}", x="", y=y_label, fill="Class")
                + theme_bw()
                + theme(figure_size=(12, 7), axis_text_y=element_text(size=9))
            )
            if not class_pending.empty:
                metric_plot = metric_plot + geom_text(
                    data=class_pending,
                    mapping=aes(x="run_display", y="y", label="label"),
                    inherit_aes=False,
                    color="#6c757d",
                    size=9,
                )

            save_plot_multi(
                metric_plot,
                figures_dir=figures_dir,
                base_name=base_name,
                width=12,
                height=7,
            )

            class_no_probe_df = class_plot_df[~class_plot_df["is_probe"].fillna(False)].copy()
            if not class_no_probe_df.empty:
                class_no_probe_df = apply_run_order(class_no_probe_df, no_probe_order)
                no_probe_class_pending = pending_annotation_df(no_probe_runs, no_probe_order, y_value=0.03)
                if metric_col == "recall":
                    short_label_map = build_short_run_label_map(no_probe_runs, include_family=False)
                    class_no_probe_df["model_panel"] = class_no_probe_df["model_family"].map(
                        {"llama": "Llama", "gemma": "Gemma"}
                    ).fillna(class_no_probe_df["model_family"])
                    class_no_probe_df["model_panel"] = pd.Categorical(
                        class_no_probe_df["model_panel"],
                        categories=["Llama", "Gemma"],
                        ordered=True,
                    )
                    metric_no_probe_plot = (
                        ggplot(class_no_probe_df, aes(x="run_display", y=metric_col, fill="behavior_class_display"))
                        + geom_col(position=position_dodge(width=0.78), width=0.7)
                        + coord_flip()
                        + facet_wrap("~model_panel", ncol=1, scales="free_y")
                        + scale_x_discrete(labels=short_label_map)
                        + scale_fill_manual(values=PREDICTION_COLORS)
                        + scale_y_continuous(
                            labels=percent_format(),
                            limits=(0.0, 1.0),
                            breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        )
                        + labs(
                            title=f"{title_suffix} on When2Call {no_probe_suffix}",
                            x="",
                            y=y_label,
                            fill="Class",
                        )
                        + theme_bw(base_size=18)
                        + theme(
                            figure_size=(9, 18),
                            axis_text_y=element_text(size=18),
                            axis_text_x=element_text(size=17),
                            strip_text=element_text(size=18),
                            legend_position="bottom",
                            legend_title=element_text(size=16),
                            legend_text=element_text(size=15),
                            plot_title=element_text(size=20),
                        )
                    )
                    if not no_probe_class_pending.empty:
                        panel_map = no_probe_runs[["run_display", "model_family"]].drop_duplicates().copy()
                        panel_map["model_panel"] = panel_map["model_family"].map(
                            {"llama": "Llama", "gemma": "Gemma"}
                        ).fillna(panel_map["model_family"])
                        pending_for_facets = no_probe_class_pending.merge(
                            panel_map[["run_display", "model_panel"]],
                            on="run_display",
                            how="left",
                        )
                        metric_no_probe_plot = metric_no_probe_plot + geom_text(
                            data=pending_for_facets,
                            mapping=aes(x="run_display", y="y", label="label"),
                            inherit_aes=False,
                            color="#6c757d",
                            size=12,
                        )
                    save_plot_multi(
                        metric_no_probe_plot,
                        figures_dir=figures_dir,
                        base_name=f"{base_name}_without_probe",
                        width=9,
                        height=18,
                    )
                else:
                    metric_no_probe_plot = (
                        ggplot(class_no_probe_df, aes(x="run_display", y=metric_col, fill="behavior_class_display"))
                        + geom_col(position=position_dodge(width=0.78), width=0.7)
                        + coord_flip()
                        + scale_fill_manual(values=PREDICTION_COLORS)
                        + scale_y_continuous(
                            labels=percent_format(),
                            limits=(0.0, 1.0),
                            breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        )
                        + labs(
                            title=f"{title_suffix} on When2Call {no_probe_suffix}",
                            x="",
                            y=y_label,
                            fill="Class",
                        )
                        + theme_bw()
                        + theme(figure_size=(12, 7), axis_text_y=element_text(size=9))
                    )
                    if not no_probe_class_pending.empty:
                        metric_no_probe_plot = metric_no_probe_plot + geom_text(
                            data=no_probe_class_pending,
                            mapping=aes(x="run_display", y="y", label="label"),
                            inherit_aes=False,
                            color="#6c757d",
                            size=9,
                        )
                    save_plot_multi(
                        metric_no_probe_plot,
                        figures_dir=figures_dir,
                        base_name=f"{base_name}_without_probe",
                        width=12,
                        height=7,
                    )

    if not direct_df.empty:
        direct_plot_df = apply_run_order(direct_df, run_order)
        direct_pending = pending_annotation_df(runs_plot, run_order, y_value=0.03)

        direct_rate_plot = (
            ggplot(direct_plot_df, aes(x="run_display", y="direct_prediction_rate", fill="scoring"))
            + geom_col(position=position_dodge(width=0.75), width=0.68)
            + coord_flip()
            + scale_fill_manual(values=SCORING_COLORS)
            + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            + labs(
                title=f"Unsupported 'direct' Prediction Rate on When2Call {all_runs_suffix}",
                x="",
                y="Rate",
                fill="Scoring",
            )
            + theme_bw()
            + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
        )
        if not direct_pending.empty:
            direct_rate_plot = direct_rate_plot + geom_text(
                data=direct_pending,
                mapping=aes(x="run_display", y="y", label="label"),
                inherit_aes=False,
                color="#6c757d",
                size=9,
            )

        save_plot_multi(
            direct_rate_plot,
            figures_dir=figures_dir,
            base_name="unsupported_direct_prediction_rate_by_run",
            width=11,
            height=6,
        )

        direct_no_probe_df = direct_df[~direct_df["is_probe"].fillna(False)].copy()
        if not direct_no_probe_df.empty:
            direct_no_probe_df = apply_run_order(direct_no_probe_df, no_probe_order)
            no_probe_direct_pending = pending_annotation_df(no_probe_runs, no_probe_order, y_value=0.03)
            no_probe_direct_rate_plot = (
                ggplot(direct_no_probe_df, aes(x="run_display", y="direct_prediction_rate", fill="scoring"))
                + geom_col(position=position_dodge(width=0.75), width=0.68)
                + coord_flip()
                + scale_fill_manual(values=SCORING_COLORS)
                + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                + labs(
                    title=f"Unsupported 'direct' Prediction Rate on When2Call {no_probe_suffix}",
                    x="",
                    y="Rate",
                    fill="Scoring",
                )
                + theme_bw()
                + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
            )
            if not no_probe_direct_pending.empty:
                no_probe_direct_rate_plot = no_probe_direct_rate_plot + geom_text(
                    data=no_probe_direct_pending,
                    mapping=aes(x="run_display", y="y", label="label"),
                    inherit_aes=False,
                    color="#6c757d",
                    size=9,
                )
            save_plot_multi(
                no_probe_direct_rate_plot,
                figures_dir=figures_dir,
                base_name="unsupported_direct_prediction_rate_by_run_without_probe",
                width=11,
                height=6,
            )

    if not outcome_df.empty:
        outcome_plot_df = apply_run_order(outcome_df, run_order)
        outcome_pending = pending_annotation_df(runs_plot, run_order, y_value=0.05)

        outcome_plot = (
            ggplot(outcome_plot_df, aes(x="run_display", y="fraction", fill="outcome"))
            + geom_col(width=0.75)
            + coord_flip()
            + scale_fill_manual(values=OUTCOME_COLORS)
            + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            + labs(
                title=f"Normalization Outcome Breakdown on When2Call {all_runs_suffix}",
                x="",
                y="Fraction of Examples",
                fill="Outcome",
            )
            + theme_bw()
            + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
        )
        if not outcome_pending.empty:
            outcome_plot = outcome_plot + geom_text(
                data=outcome_pending,
                mapping=aes(x="run_display", y="y", label="label"),
                inherit_aes=False,
                color="#6c757d",
                size=9,
            )

        save_plot_multi(
            outcome_plot,
            figures_dir=figures_dir,
            base_name="normalization_outcome_breakdown_by_run",
            width=11,
            height=6,
        )

        outcome_no_probe_df = outcome_df[~outcome_df["is_probe"].fillna(False)].copy()
        if not outcome_no_probe_df.empty:
            outcome_no_probe_df = apply_run_order(outcome_no_probe_df, no_probe_order)
            no_probe_outcome_pending = pending_annotation_df(no_probe_runs, no_probe_order, y_value=0.05)
            no_probe_outcome_plot = (
                ggplot(outcome_no_probe_df, aes(x="run_display", y="fraction", fill="outcome"))
                + geom_col(width=0.75)
                + coord_flip()
                + scale_fill_manual(values=OUTCOME_COLORS)
                + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                + labs(
                    title=f"Normalization Outcome Breakdown on When2Call {no_probe_suffix}",
                    x="",
                    y="Fraction of Examples",
                    fill="Outcome",
                )
                + theme_bw()
                + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
            )
            if not no_probe_outcome_pending.empty:
                no_probe_outcome_plot = no_probe_outcome_plot + geom_text(
                    data=no_probe_outcome_pending,
                    mapping=aes(x="run_display", y="y", label="label"),
                    inherit_aes=False,
                    color="#6c757d",
                    size=9,
                )
            save_plot_multi(
                no_probe_outcome_plot,
                figures_dir=figures_dir,
                base_name="normalization_outcome_breakdown_by_run_without_probe",
                width=11,
                height=6,
            )

    if not prediction_mix_df.empty:
        pred_mix_df = prediction_mix_df[prediction_mix_df["scoring"] == "normalized"].copy()
        if not pred_mix_df.empty:
            pred_mix_df = apply_run_order(pred_mix_df, run_order)
            pred_mix_df["label_display"] = pred_mix_df["label"].map(CLASS_DISPLAY).fillna(pred_mix_df["label"])
            pred_pending = pending_annotation_df(runs_plot, run_order, y_value=0.05)

            pred_mix_plot = (
                ggplot(pred_mix_df, aes(x="run_display", y="fraction", fill="label_display"))
                + geom_col(width=0.75)
                + coord_flip()
                + scale_fill_manual(values=PREDICTION_COLORS)
                + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.0), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                + labs(
                    title=f"Prediction Mix on When2Call {all_runs_suffix}",
                    x="",
                    y="Fraction of Predictions",
                    fill="Predicted Label",
                )
                + theme_bw()
                + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
            )
            if not pred_pending.empty:
                pred_mix_plot = pred_mix_plot + geom_text(
                    data=pred_pending,
                    mapping=aes(x="run_display", y="y", label="label"),
                    inherit_aes=False,
                    color="#6c757d",
                    size=9,
                )

            save_plot_multi(
                pred_mix_plot,
                figures_dir=figures_dir,
                base_name="normalized_prediction_mix_by_run",
                width=11,
                height=6,
            )

            no_probe_pred_mix = pred_mix_df[~pred_mix_df["is_probe"].fillna(False)].copy()
            if not no_probe_pred_mix.empty:
                no_probe_pred_mix = apply_run_order(no_probe_pred_mix, no_probe_order)
                no_probe_pending = pending_annotation_df(
                    no_probe_runs,
                    no_probe_order,
                    y_value=0.05,
                )

                no_probe_mix_plot = (
                    ggplot(no_probe_pred_mix, aes(x="run_display", y="fraction", fill="label_display"))
                    + geom_col(width=0.75)
                    + coord_flip()
                    + scale_fill_manual(values=PREDICTION_COLORS)
                    + scale_y_continuous(
                        labels=percent_format(),
                        limits=(0.0, 1.0),
                        breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    )
                    + labs(
                        title=f"Prediction Mix on When2Call {no_probe_suffix}",
                        x="",
                        y="Fraction of Predictions",
                        fill="Predicted Label",
                    )
                    + theme_bw()
                    + theme(figure_size=(11, 6), axis_text_y=element_text(size=9))
                )
                if not no_probe_pending.empty:
                    no_probe_mix_plot = no_probe_mix_plot + geom_text(
                        data=no_probe_pending,
                        mapping=aes(x="run_display", y="y", label="label"),
                        inherit_aes=False,
                        color="#6c757d",
                        size=9,
                    )

                save_plot_multi(
                    no_probe_mix_plot,
                    figures_dir=figures_dir,
                    base_name="normalized_prediction_mix_by_run_without_probe",
                    width=11,
                    height=6,
                )


def make_probe_only_plots(*, probe_layers_df: pd.DataFrame, figures_dir: Path) -> None:
    if probe_layers_df.empty:
        return

    probe_plot = sort_runs(probe_layers_df)
    parent_order = probe_plot["probe_parent_display"].drop_duplicates().tolist()
    probe_plot["probe_parent_display"] = pd.Categorical(
        probe_plot["probe_parent_display"],
        categories=parent_order,
        ordered=True,
    )
    probe_plot["layer_display"] = pd.Categorical(
        probe_plot["variant_label"],
        categories=["Middle layer", "75% depth layer", "Last layer"],
        ordered=True,
    )
    pending_parent = (
        probe_plot.groupby("probe_parent_display", observed=False)["is_placeholder"]
        .all()
        .reset_index()
    )
    pending_parent = pending_parent[pending_parent["is_placeholder"]].copy()
    if not pending_parent.empty:
        pending_parent["y"] = 0.04
        pending_parent["label"] = "PENDING"

    def _plot_probe_metric(metric_col: str, y_label: str, title: str, base_name: str) -> None:
        metric_plot = (
            ggplot(probe_plot, aes(x="probe_parent_display", y=metric_col, fill="layer_display"))
            + geom_col(position=position_dodge(width=0.78), width=0.7)
            + coord_flip()
            + scale_fill_manual(values=PROBE_LAYER_COLORS)
            + scale_y_continuous(labels=percent_format(), limits=(0.0, 1.12), breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            + labs(title=title, x="", y=y_label, fill="Layer")
            + theme_bw()
            + theme(figure_size=(12, 7), axis_text_y=element_text(size=9))
        )
        if not pending_parent.empty:
            metric_plot = metric_plot + geom_text(
                data=pending_parent,
                mapping=aes(x="probe_parent_display", y="y", label="label"),
                inherit_aes=False,
                color="#6c757d",
                size=8,
            )
        save_plot_multi(
            metric_plot,
            figures_dir=figures_dir,
            base_name=base_name,
            width=12,
            height=7,
        )

    _plot_probe_metric(
        metric_col="norm_accuracy",
        y_label="Accuracy",
        title="Probe Layer Accuracy on When2Call",
        base_name="probe_layer_accuracy_comparison",
    )
    _plot_probe_metric(
        metric_col="norm_macro_f1",
        y_label="Macro-F1",
        title="Probe Layer Macro-F1 on When2Call",
        base_name="probe_layer_macro_f1_comparison",
    )
    _plot_probe_metric(
        metric_col="norm_macro_precision",
        y_label="Macro Precision",
        title="Probe Layer Macro Precision on When2Call",
        base_name="probe_layer_macro_precision_comparison",
    )
    _plot_probe_metric(
        metric_col="norm_macro_recall",
        y_label="Macro Recall",
        title="Probe Layer Macro Recall on When2Call",
        base_name="probe_layer_macro_recall_comparison",
    )
    _plot_probe_metric(
        metric_col="agreement_with_model_pred_norm",
        y_label="Agreement",
        title="Probe-Model Agreement (Model Normalized Pred) on When2Call",
        base_name="probe_layer_model_agreement_norm_comparison",
    )
    _plot_probe_metric(
        metric_col="agreement_with_model_pred_raw",
        y_label="Agreement",
        title="Probe-Model Agreement (Model Raw Pred) on When2Call",
        base_name="probe_layer_model_agreement_raw_comparison",
    )


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    probe_runs_dir = Path(args.probe_runs_dir).resolve()
    if not probe_runs_dir.exists() and (runs_dir / "Probe Evals").exists():
        probe_runs_dir = (runs_dir / "Probe Evals").resolve()

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    data_dir = ensure_dir(output_dir / "data")
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    run_dirs = discover_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No eval run directories found under {runs_dir}.")
    probe_dirs = discover_probe_dirs(probe_runs_dir)

    run_records: List[Dict[str, Any]] = []
    class_records: List[Dict[str, Any]] = []
    direct_records: List[Dict[str, Any]] = []
    outcome_records: List[Dict[str, Any]] = []
    prediction_mix_records: List[Dict[str, Any]] = []
    probe_layer_records: List[Dict[str, Any]] = []

    for run_dir, source_group in run_dirs:
        run_key = run_dir.name
        summary = load_json(run_dir / "summary.json")
        run_config = load_json(run_dir / "run_config.json")

        model_family = infer_model_family(run_key, run_config)
        run_group = infer_run_group(run_key, run_config, source_group=source_group)
        variant_label = infer_eval_variant_label(run_key, run_config, run_group=run_group)
        _, family_short = family_display_names(model_family)
        run_display = make_run_display(family_short, run_group, variant_label)

        run_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": "n/a",
                "is_probe": False,
                "is_placeholder": False,
                "probe_eval_setting": "n/a",
                "raw_accuracy": float(summary.get("raw", {}).get("accuracy", 0.0)),
                "norm_accuracy": float(summary.get("normalized", {}).get("accuracy", 0.0)),
                "raw_macro_f1": float(summary.get("raw", {}).get("macro_f1", 0.0)),
                "norm_macro_f1": float(summary.get("normalized", {}).get("macro_f1", 0.0)),
                "n_examples": int(summary.get("n_examples", 0)),
            }
        )

        class_records.extend(
            extract_class_metrics(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                run_group=run_group,
                variant_label=variant_label,
                probe_source_variant="n/a",
                summary=summary,
                is_probe=False,
                is_placeholder=False,
            )
        )

        sample_path = run_dir / "samples.jsonl"
        if sample_path.exists():
            samples = read_jsonl(sample_path)
            label_order = list(summary.get("label_order") or DEFAULT_LABEL_ORDER)
            direct_rows, outcome_rows, pred_mix_rows = extract_sample_level_summaries(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                run_group=run_group,
                variant_label=variant_label,
                probe_source_variant="n/a",
                samples=samples,
                label_order=label_order,
                is_probe=False,
                is_placeholder=False,
            )
            direct_records.extend(direct_rows)
            outcome_records.extend(outcome_rows)
            prediction_mix_records.extend(pred_mix_rows)

    for probe_dir in probe_dirs:
        probe_eval = load_json(probe_dir / "probe_evaluation_summary.json")
        run_config = load_json(probe_dir / "run_config.json")

        model_family = infer_model_family(probe_dir.name, run_config)
        run_group = "Probe"
        probe_eval_setting = infer_probe_eval_setting(probe_dir.name, run_config)
        probe_source_variant = infer_probe_source_variant(probe_dir.name, run_config)
        _, family_short = family_display_names(model_family)
        probe_parent_display = make_probe_parent_display(family_short, probe_eval_setting, probe_source_variant)

        layers: Dict[str, Dict[str, Any]] = probe_eval.get("layers") or {}
        layer_order = [
            str(spec.get("tag"))
            for spec in (probe_eval.get("layer_specs") or [])
            if str(spec.get("tag") or "").strip()
        ]
        if not layer_order:
            layer_order = sorted(layers.keys())

        probe_samples_path = probe_dir / "probe_comparison_samples.jsonl"
        probe_samples = read_jsonl(probe_samples_path) if probe_samples_path.exists() else []

        candidate_layer_tags = [tag for tag in layer_order if tag in layers]
        if not candidate_layer_tags:
            candidate_layer_tags = sorted(layers.keys())
        if not candidate_layer_tags:
            continue

        probe_is_pending = probe_source_variant in CORRUPTED_PROBE_SOURCE_VARIANTS
        if probe_is_pending:
            middle_tag = next(
                (tag for tag in candidate_layer_tags if infer_probe_variant_label(tag) == "Middle layer"),
                candidate_layer_tags[0],
            )
            best_layer_tag = middle_tag
        else:
            best_layer_tag = select_best_probe_layer(layers, candidate_layer_tags)
            if best_layer_tag is None:
                continue

        n_examples = int(probe_eval.get("num_joined_examples", 0) or 0)

        for layer_tag in candidate_layer_tags:
            layer_payload = layers.get(layer_tag) or {}
            probe_vs_gold = layer_payload.get("probe_vs_gold") or {}
            if not probe_vs_gold and not probe_is_pending:
                continue
            layer_variant = infer_probe_variant_label(layer_tag)
            layer_run_key = f"{probe_dir.name}:{layer_tag}"
            layer_run_display = make_probe_run_display(
                family_short,
                probe_eval_setting,
                layer_variant,
                probe_source_variant=probe_source_variant,
            )
            if probe_is_pending:
                layer_accuracy = 0.0
                layer_macro_f1 = 0.0
                layer_macro_precision = 0.0
                layer_macro_recall = 0.0
                agreement_norm = 0.0
                agreement_raw = 0.0
                agreement_norm_restricted = 0.0
                agreement_raw_restricted = 0.0
            else:
                layer_accuracy, layer_macro_f1 = probe_layer_score(layer_payload)
                class_report = probe_vs_gold.get("classification_report", {}) or {}
                macro_avg = class_report.get("macro avg", {}) or {}
                layer_macro_precision = float(
                    probe_vs_gold.get("macro_precision", macro_avg.get("precision", 0.0)) or 0.0
                )
                layer_macro_recall = float(
                    probe_vs_gold.get("macro_recall", macro_avg.get("recall", 0.0)) or 0.0
                )
                probe_vs_model_norm = layer_payload.get("probe_vs_model_pred_norm", {}) or {}
                probe_vs_model_raw = layer_payload.get("probe_vs_model_pred_raw", {}) or {}
                agreement_norm = float(probe_vs_model_norm.get("overall_agreement", 0.0) or 0.0)
                agreement_raw = float(probe_vs_model_raw.get("overall_agreement", 0.0) or 0.0)
                agreement_norm_restricted = float(
                    ((probe_vs_model_norm.get("restricted_to_probe_label_space") or {}).get("accuracy", 0.0) or 0.0)
                )
                agreement_raw_restricted = float(
                    ((probe_vs_model_raw.get("restricted_to_probe_label_space") or {}).get("accuracy", 0.0) or 0.0)
                )
            probe_layer_records.append(
                {
                    "run_key": layer_run_key,
                    "run_display": layer_run_display,
                    "probe_parent_key": probe_dir.name,
                    "probe_parent_display": probe_parent_display,
                    "model_family": model_family,
                    "run_group": run_group,
                    "variant_label": layer_variant,
                    "probe_source_variant": probe_source_variant,
                    "is_probe": True,
                    "is_placeholder": probe_is_pending,
                    "probe_eval_setting": probe_eval_setting,
                    "raw_accuracy": layer_accuracy,
                    "norm_accuracy": layer_accuracy,
                    "raw_macro_f1": layer_macro_f1,
                    "norm_macro_f1": layer_macro_f1,
                    "raw_macro_precision": layer_macro_precision,
                    "norm_macro_precision": layer_macro_precision,
                    "raw_macro_recall": layer_macro_recall,
                    "norm_macro_recall": layer_macro_recall,
                    "agreement_with_model_pred_norm": agreement_norm,
                    "agreement_with_model_pred_raw": agreement_raw,
                    "agreement_with_model_pred_norm_restricted": agreement_norm_restricted,
                    "agreement_with_model_pred_raw_restricted": agreement_raw_restricted,
                    "n_examples": n_examples,
                    "used_in_main_plots": bool(layer_tag == best_layer_tag),
                    "accuracy_rank_within_probe": 0,
                    "macro_f1_rank_within_probe": 0,
                    "selection_reason": "",
                }
            )

        layer_payload = layers.get(best_layer_tag) or {}
        probe_vs_gold = layer_payload.get("probe_vs_gold") or {}
        if not probe_vs_gold and not probe_is_pending:
            continue

        variant_label = infer_probe_variant_label(best_layer_tag)
        run_key = f"{probe_dir.name}:{best_layer_tag}"
        run_display = make_probe_run_display(
            family_short,
            probe_eval_setting,
            variant_label,
            probe_source_variant=probe_source_variant,
        )
        if probe_is_pending:
            probe_accuracy = 0.0
            probe_macro_f1 = 0.0
        else:
            probe_accuracy, probe_macro_f1 = probe_layer_score(layer_payload)

        run_records.append(
            {
                "run_key": run_key,
                "run_display": run_display,
                "model_family": model_family,
                "run_group": run_group,
                "variant_label": variant_label,
                "probe_source_variant": probe_source_variant,
                "is_probe": True,
                "is_placeholder": probe_is_pending,
                "probe_eval_setting": probe_eval_setting,
                "raw_accuracy": probe_accuracy,
                "norm_accuracy": probe_accuracy,
                "raw_macro_f1": probe_macro_f1,
                "norm_macro_f1": probe_macro_f1,
                "n_examples": n_examples,
            }
        )

        if probe_is_pending:
            class_rows, direct_rows, outcome_rows, pred_mix_rows = build_probe_pending_rows(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                run_group=run_group,
                variant_label=variant_label,
                probe_source_variant=probe_source_variant,
            )
            class_records.extend(class_rows)
            direct_records.extend(direct_rows)
            outcome_records.extend(outcome_rows)
            prediction_mix_records.extend(pred_mix_rows)
            continue

        summary_like = build_probe_summary_like(probe_vs_gold, n_examples)
        class_records.extend(
            extract_class_metrics(
                run_key=run_key,
                run_display=run_display,
                model_family=model_family,
                run_group=run_group,
                variant_label=variant_label,
                probe_source_variant=probe_source_variant,
                summary=summary_like,
                is_probe=True,
                is_placeholder=False,
            )
        )

        if probe_samples:
            layer_samples: List[Dict[str, Any]] = []
            for sample in probe_samples:
                probe_pred = (sample.get("probe_preds") or {}).get(best_layer_tag)
                if probe_pred is None:
                    continue
                layer_samples.append(
                    {
                        "gold": sample.get("gold"),
                        "pred_raw": probe_pred,
                        "pred_norm": probe_pred,
                    }
                )
            if layer_samples:
                direct_rows, outcome_rows, pred_mix_rows = extract_sample_level_summaries(
                    run_key=run_key,
                    run_display=run_display,
                    model_family=model_family,
                    run_group=run_group,
                    variant_label=variant_label,
                    probe_source_variant=probe_source_variant,
                    samples=layer_samples,
                    label_order=summary_like.get("label_order") or DEFAULT_LABEL_ORDER,
                    is_probe=True,
                    is_placeholder=False,
                )
                direct_records.extend(direct_rows)
                outcome_records.extend(outcome_rows)
                prediction_mix_records.extend(pred_mix_rows)

    runs_df = pd.DataFrame(run_records, columns=RUN_COLUMNS)
    class_df = pd.DataFrame(class_records, columns=CLASS_COLUMNS)
    direct_df = pd.DataFrame(direct_records, columns=DIRECT_COLUMNS)
    outcome_df = pd.DataFrame(outcome_records, columns=OUTCOME_COLUMNS)
    prediction_mix_df = pd.DataFrame(prediction_mix_records, columns=PRED_MIX_COLUMNS)
    probe_layers_df = pd.DataFrame(probe_layer_records, columns=PROBE_LAYER_COLUMNS)
    if not probe_layers_df.empty:
        probe_layers_df = enrich_probe_layer_selection(probe_layers_df)

    missing_placeholder_keys: List[str] = []
    if not args.no_placeholders:
        missing_placeholder_keys = infer_missing_gemma_cai_eval_run_keys(runs_df["run_key"].tolist())
        placeholder_run_rows, placeholder_class_rows, placeholder_direct_rows, placeholder_outcome_rows, placeholder_predmix_rows = build_placeholder_rows(
            missing_placeholder_keys
        )
        if placeholder_run_rows:
            runs_df = pd.concat([runs_df, pd.DataFrame(placeholder_run_rows, columns=RUN_COLUMNS)], ignore_index=True)
        if placeholder_class_rows:
            class_df = pd.concat([class_df, pd.DataFrame(placeholder_class_rows, columns=CLASS_COLUMNS)], ignore_index=True)
        if placeholder_direct_rows:
            direct_df = pd.concat([direct_df, pd.DataFrame(placeholder_direct_rows, columns=DIRECT_COLUMNS)], ignore_index=True)
        if placeholder_outcome_rows:
            outcome_df = pd.concat([outcome_df, pd.DataFrame(placeholder_outcome_rows, columns=OUTCOME_COLUMNS)], ignore_index=True)
        if placeholder_predmix_rows:
            prediction_mix_df = pd.concat(
                [prediction_mix_df, pd.DataFrame(placeholder_predmix_rows, columns=PRED_MIX_COLUMNS)],
                ignore_index=True,
            )

    runs_df, class_df, direct_df, outcome_df, prediction_mix_df = enforce_unique_run_displays(
        runs_df,
        class_df,
        direct_df,
        outcome_df,
        prediction_mix_df,
    )

    runs_df = sort_runs(runs_df)
    if not class_df.empty:
        class_df = sort_runs(class_df)
    if not direct_df.empty:
        direct_df = sort_runs(direct_df)
    if not outcome_df.empty:
        outcome_df = sort_runs(outcome_df)
    if not prediction_mix_df.empty:
        prediction_mix_df = sort_runs(prediction_mix_df)
    if not probe_layers_df.empty:
        probe_layers_df = sort_runs(probe_layers_df)

    runs_df.to_csv(data_dir / "run_level_metrics.csv", index=False)
    class_df.to_csv(data_dir / "class_level_metrics.csv", index=False)
    direct_df.to_csv(data_dir / "direct_prediction_rate.csv", index=False)
    outcome_df.to_csv(data_dir / "normalization_outcomes.csv", index=False)
    prediction_mix_df.to_csv(data_dir / "prediction_mix.csv", index=False)
    probe_layers_df.to_csv(data_dir / "probe_layer_metrics.csv", index=False)
    if not probe_layers_df.empty:
        probe_best_df = probe_layers_df[probe_layers_df["used_in_main_plots"].fillna(False)].copy()
        probe_best_df = probe_best_df[
            [
                "probe_parent_key",
                "probe_parent_display",
                "run_key",
                "run_display",
                "variant_label",
                "norm_accuracy",
                "norm_macro_f1",
                "selection_reason",
            ]
        ]
        probe_best_df.to_csv(data_dir / "probe_best_layer_selection.csv", index=False)

    make_bar_plots(
        runs_df=runs_df,
        class_df=class_df,
        direct_df=direct_df,
        outcome_df=outcome_df,
        prediction_mix_df=prediction_mix_df,
        figures_dir=figures_dir,
    )
    make_probe_only_plots(
        probe_layers_df=probe_layers_df,
        figures_dir=figures_dir,
    )
    write_chart_tables(
        runs_df=runs_df,
        class_df=class_df,
        direct_df=direct_df,
        outcome_df=outcome_df,
        prediction_mix_df=prediction_mix_df,
        probe_layers_df=probe_layers_df,
        tables_dir=tables_dir,
    )

    write_summary_markdown(
        runs_df=runs_df,
        missing_placeholder_keys=missing_placeholder_keys,
        summary_path=output_dir / "analysis_summary.md",
    )

    print(
        json.dumps(
            {
                "runs_analyzed": int(len(runs_df)),
                "probe_layer_rows": int(len(probe_layers_df)),
                "placeholder_runs": int(len(missing_placeholder_keys)),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
