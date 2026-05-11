#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence


ACTIVE_LABELS = ["tool_call", "request_for_info", "cannot_answer"]
LABEL_DISPLAY = {
    "tool_call": "Tool Call",
    "request_for_info": "Request Info",
    "cannot_answer": "Cannot Answer",
}
LABEL_LINES = {
    "tool_call": ["Tool", "Call"],
    "request_for_info": ["Request", "Info"],
    "cannot_answer": ["Cannot", "Answer"],
}
RAW_BAR_COLOR = "#cbd5e1"
NORM_BAR_COLOR = "#0f766e"
OUTCOME_COLORS = {
    "stay_correct": "#15803d",
    "fixed": "#2563eb",
    "broken": "#dc2626",
    "stay_wrong": "#94a3b8",
}
METRIC_SPECS = [
    (
        "accuracy",
        "Accuracy",
        "Overall exact-match accuracy. Each example counts once.",
    ),
    (
        "precision",
        "Precision",
        "Weighted-average precision across labels from classification_report.",
    ),
    (
        "recall",
        "Recall",
        "Weighted-average recall across labels from classification_report.",
    ),
    (
        "f1",
        "F1",
        "Weighted-average F1 across labels from classification_report.",
    ),
    (
        "macro_f1",
        "Macro-F1",
        "Unweighted mean of per-class F1 across the evaluator label order.",
    ),
]
METHOD_ORDER = {
    "0-shot": 0,
    "4-shot": 1,
    "SFT": 2,
    "DPO": 3,
    "CAI-SFT Self": 4,
    "CAI-SFT Cross": 5,
    "CAI-DPO Self": 6,
    "CAI-DPO Cross": 7,
}
FAMILY_ORDER = {
    "llama": 0,
    "gemma": 1,
}
RUN_PALETTE = {
    ("llama", "0-shot"): "#0f766e",
    ("llama", "4-shot"): "#14b8a6",
    ("llama", "SFT"): "#2dd4bf",
    ("llama", "DPO"): "#0891b2",
    ("llama", "CAI-SFT Self"): "#7c3aed",
    ("llama", "CAI-SFT Cross"): "#a855f7",
    ("llama", "CAI-DPO Self"): "#4f46e5",
    ("llama", "CAI-DPO Cross"): "#6366f1",
    ("gemma", "0-shot"): "#b45309",
    ("gemma", "4-shot"): "#f59e0b",
    ("gemma", "SFT"): "#fb923c",
    ("gemma", "DPO"): "#ea580c",
    ("gemma", "CAI-SFT Self"): "#7c2d12",
    ("gemma", "CAI-SFT Cross"): "#9a3412",
    ("gemma", "CAI-DPO Self"): "#991b1b",
    ("gemma", "CAI-DPO Cross"): "#b91c1c",
}


@dataclass
class AggregateMetric:
    raw: float
    normalized: float


@dataclass
class RunMetrics:
    run_key: str
    display_name: str
    legend_name: str
    label_lines: List[str]
    model_family: str
    variant_label: str
    summary: Dict
    samples: List[Dict]
    aggregate_metrics: Dict[str, AggregateMetric]
    normalized_per_class_f1: Dict[str, float]
    normalized_per_class_accuracy: Dict[str, float]
    normalized_per_class_precision: Dict[str, float]
    normalized_per_class_recall: Dict[str, float]
    normalized_confusion_active: List[List[int]]
    gold_support: Dict[str, int]
    normalization_outcomes: Dict[str, int]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def family_names(model_name_or_path: str, model_family: str | None = None) -> tuple[str, str, str]:
    family_hint = (model_family or "").lower()
    model_name = model_name_or_path.lower()
    for key, family_name, short_family in (
        ("llama", "Llama 3.2 3B", "Llama"),
        ("gemma", "Gemma 3 4B", "Gemma"),
    ):
        if family_hint == key or key in model_name:
            return family_name, short_family, key
    return model_name_or_path, model_name_or_path, model_name_or_path.lower()


def detect_variant(run_key: str, run_config: Dict) -> str:
    haystack = " ".join(
        [
            run_key.lower(),
            str(run_config.get("output_dir", "")).lower(),
            str(run_config.get("model_name_or_path", "")).lower(),
        ]
    )
    if "cai" in haystack and "dpo" in haystack and "cross" in haystack:
        return "CAI-DPO Cross"
    if "cai" in haystack and "dpo" in haystack and "self" in haystack:
        return "CAI-DPO Self"
    if "cai" in haystack and "sft" in haystack and "cross" in haystack:
        return "CAI-SFT Cross"
    if "cai" in haystack and "sft" in haystack and "self" in haystack:
        return "CAI-SFT Self"
    if "dpo" in haystack:
        return "DPO"
    if "sft" in haystack:
        return "SFT"
    num_shots = int(run_config.get("num_shots", 0))
    if "4shot" in haystack or num_shots == 4:
        return "4-shot"
    return "0-shot"


def slug_to_display_name(run_key: str, run_config: Dict) -> tuple[str, str, List[str], str, str]:
    family_name, short_family, family_key = family_names(
        str(run_config["model_name_or_path"]),
        str(run_config.get("model_family", "")),
    )
    variant_label = detect_variant(run_key, run_config)
    return (
        f"{family_name} {variant_label}",
        f"{short_family} {variant_label}",
        [family_name, variant_label],
        family_key,
        variant_label,
    )


def discover_run_dirs(base_dir: Path) -> List[Path]:
    required_files = {"summary.json", "samples.jsonl", "run_config.json"}
    run_dirs = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if required_files.issubset({path.name for path in child.iterdir() if path.is_file()}):
            run_dirs.append(child)
    return run_dirs


def run_sort_key(run: RunMetrics) -> tuple[int, int, str]:
    family_rank = FAMILY_ORDER.get(run.model_family, 99)
    method_rank = METHOD_ORDER.get(run.variant_label, 99)
    return (family_rank, method_rank, run.display_name)


def run_color(run: RunMetrics) -> str:
    return RUN_PALETTE.get((run.model_family, run.variant_label), "#64748b")


def compute_normalization_outcomes(samples: Sequence[Dict]) -> Dict[str, int]:
    outcomes = {
        "stay_correct": 0,
        "fixed": 0,
        "broken": 0,
        "stay_wrong": 0,
    }
    for sample in samples:
        raw_correct = sample["pred_raw"] == sample["gold"]
        norm_correct = sample["pred_norm"] == sample["gold"]
        if raw_correct and norm_correct:
            outcomes["stay_correct"] += 1
        elif not raw_correct and norm_correct:
            outcomes["fixed"] += 1
        elif raw_correct and not norm_correct:
            outcomes["broken"] += 1
        else:
            outcomes["stay_wrong"] += 1
    return outcomes


def aggregate_metrics_from_summary(summary: Dict) -> Dict[str, AggregateMetric]:
    raw_report = summary["raw"]["classification_report"]
    norm_report = summary["normalized"]["classification_report"]
    raw_weighted = raw_report["weighted avg"]
    norm_weighted = norm_report["weighted avg"]
    return {
        "accuracy": AggregateMetric(
            raw=float(summary["raw"]["accuracy"]),
            normalized=float(summary["normalized"]["accuracy"]),
        ),
        "precision": AggregateMetric(
            raw=float(raw_weighted["precision"]),
            normalized=float(norm_weighted["precision"]),
        ),
        "recall": AggregateMetric(
            raw=float(raw_weighted["recall"]),
            normalized=float(norm_weighted["recall"]),
        ),
        "f1": AggregateMetric(
            raw=float(raw_weighted["f1-score"]),
            normalized=float(norm_weighted["f1-score"]),
        ),
        "macro_f1": AggregateMetric(
            raw=float(summary["raw"]["macro_f1"]),
            normalized=float(summary["normalized"]["macro_f1"]),
        ),
    }


def load_runs(base_dir: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []
    for run_dir in discover_run_dirs(base_dir):
        run_key = run_dir.name
        summary_path = run_dir / "summary.json"
        samples_path = run_dir / "samples.jsonl"
        config_path = run_dir / "run_config.json"

        summary = load_json(summary_path)
        samples = load_jsonl(samples_path)
        config = load_json(config_path)
        display_name, legend_name, label_lines, family_key, variant_label = slug_to_display_name(
            run_key, config
        )

        label_order = summary["label_order"]
        norm_report = summary["normalized"]["classification_report"]
        confusion = summary["normalized"]["confusion_matrix"]
        active_indices = [label_order.index(label) for label in ACTIVE_LABELS]
        active_confusion = [
            [int(confusion[row_index][col_index]) for col_index in active_indices]
            for row_index in active_indices
        ]
        gold_support = {
            label: int(norm_report[label]["support"])
            for label in label_order
        }
        normalized_per_class_accuracy = {}
        normalized_per_class_f1 = {}
        normalized_per_class_precision = {}
        normalized_per_class_recall = {}
        for row_idx, label in enumerate(ACTIVE_LABELS):
            support = gold_support[label]
            correct = active_confusion[row_idx][row_idx]
            normalized_per_class_accuracy[label] = 0.0 if support == 0 else correct / support
            normalized_per_class_f1[label] = float(norm_report[label]["f1-score"])
            normalized_per_class_precision[label] = float(norm_report[label]["precision"])
            normalized_per_class_recall[label] = float(norm_report[label]["recall"])

        runs.append(
            RunMetrics(
                run_key=run_key,
                display_name=display_name,
                legend_name=legend_name,
                label_lines=label_lines,
                model_family=family_key,
                variant_label=variant_label,
                summary=summary,
                samples=samples,
                aggregate_metrics=aggregate_metrics_from_summary(summary),
                normalized_per_class_f1=normalized_per_class_f1,
                normalized_per_class_accuracy=normalized_per_class_accuracy,
                normalized_per_class_precision=normalized_per_class_precision,
                normalized_per_class_recall=normalized_per_class_recall,
                normalized_confusion_active=active_confusion,
                gold_support=gold_support,
                normalization_outcomes=compute_normalization_outcomes(samples),
            )
        )

    if not runs:
        raise FileNotFoundError(
            "No prompting result directories with summary.json and samples.jsonl were found."
        )
    return sorted(runs, key=run_sort_key)


def lerp_color(start: str, end: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    s = [int(start[i : i + 2], 16) for i in (1, 3, 5)]
    e = [int(end[i : i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(sv + (ev - sv) * t) for sv, ev in zip(s, e)]
    return "#" + "".join(f"{value:02x}" for value in rgb)


def axis_max(values: Sequence[float], minimum: float = 0.4) -> float:
    upper = max(values) if values else minimum
    return min(1.0, max(minimum, math.ceil((upper + 0.05) * 10) / 10))


def per_class_metric_value(run: RunMetrics, label: str, metric_key: str) -> float:
    if metric_key == "f1":
        return run.normalized_per_class_f1[label]
    if metric_key == "accuracy":
        return run.normalized_per_class_accuracy[label]
    if metric_key == "precision":
        return run.normalized_per_class_precision[label]
    if metric_key == "recall":
        return run.normalized_per_class_recall[label]
    raise ValueError(f"Unsupported per-class metric: {metric_key}")


class SvgCanvas:
    def __init__(self, width: int, height: int, background: str = "#fffdf7"):
        self.width = width
        self.height = height
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" fill="none">',
            f'<rect width="{width}" height="{height}" fill="{background}" rx="24"/>',
            "<style>"
            "text { font-family: 'Avenir Next', 'Segoe UI', sans-serif; fill: #111827; }"
            ".title { font-size: 26px; font-weight: 700; }"
            ".subtitle { font-size: 14px; fill: #475569; }"
            ".label { font-size: 13px; }"
            ".small { font-size: 12px; }"
            ".legend { font-size: 13px; }"
            ".muted { fill: #64748b; }"
            ".grid { stroke: #e2e8f0; stroke-width: 1; }"
            ".axis { stroke: #94a3b8; stroke-width: 1.5; }"
            "</style>",
        ]

    def add(self, snippet: str) -> None:
        self.parts.append(snippet)

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        css_class: str = "",
        anchor: str = "start",
        weight: str | None = None,
        fill: str | None = None,
    ) -> None:
        attrs = [f'x="{x}"', f'y="{y}"', f'text-anchor="{anchor}"']
        if css_class:
            attrs.append(f'class="{css_class}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if fill:
            attrs.append(f'fill="{fill}"')
        self.add(f"<text {' '.join(attrs)}>{escape(content)}</text>")

    def text_lines(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        *,
        css_class: str = "",
        anchor: str = "start",
        weight: str | None = None,
        fill: str | None = None,
        line_height: float = 16,
    ) -> None:
        attrs = [f'x="{x}"', f'y="{y}"', f'text-anchor="{anchor}"']
        if css_class:
            attrs.append(f'class="{css_class}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if fill:
            attrs.append(f'fill="{fill}"')
        tspans = []
        for index, line in enumerate(lines):
            dy = "0" if index == 0 else str(line_height)
            tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.add(f"<text {' '.join(attrs)}>{''.join(tspans)}</text>")

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        rx: float = 0,
        stroke: str | None = None,
    ) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" '
            f'fill="{fill}"{stroke_attr}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, css_class: str = "grid") -> None:
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{css_class}"/>')

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def write_svg(path: Path, canvas: SvgCanvas) -> None:
    path.write_text(canvas.finish(), encoding="utf-8")


def draw_axis_grid(
    canvas: SvgCanvas,
    *,
    chart_x: float,
    chart_y: float,
    chart_width: float,
    chart_height: float,
    ymax: float,
) -> None:
    tick_step = 0.1
    num_ticks = int(round(ymax / tick_step))
    for tick in range(num_ticks + 1):
        value = tick * tick_step
        y = chart_y + chart_height - (value / ymax) * chart_height
        canvas.line(chart_x, y, chart_x + chart_width, y)
        canvas.text(chart_x - 14, y + 4, f"{value * 100:.0f}%", css_class="small muted", anchor="end")
    canvas.line(
        chart_x,
        chart_y + chart_height,
        chart_x + chart_width,
        chart_y + chart_height,
        css_class="axis",
    )


def render_metric_chart(
    runs: Sequence[RunMetrics],
    out_path: Path,
    *,
    metric_key: str,
    title: str,
    subtitle: str,
) -> None:
    run_count = len(runs)
    width = 980 + max(0, run_count - 4) * 180
    height = 740
    canvas = SvgCanvas(width, height)
    canvas.text(60, 66, title, css_class="title")
    canvas.text(60, 96, subtitle, css_class="subtitle")

    chart_x = 110
    chart_y = 182
    chart_width = width - chart_x - 80
    chart_height = 388
    values = [
        metric_value
        for run in runs
        for metric_value in (
            run.aggregate_metrics[metric_key].raw,
            run.aggregate_metrics[metric_key].normalized,
        )
    ]
    ymax = axis_max(values, minimum=0.4)
    draw_axis_grid(
        canvas,
        chart_x=chart_x,
        chart_y=chart_y,
        chart_width=chart_width,
        chart_height=chart_height,
        ymax=ymax,
    )

    group_width = chart_width / run_count
    bar_gap = 16
    bar_width = min(60.0, max(40.0, (group_width - 32 - bar_gap) / 2))
    for index, run in enumerate(runs):
        group_center = chart_x + group_width * (index + 0.5)
        raw_value = run.aggregate_metrics[metric_key].raw
        norm_value = run.aggregate_metrics[metric_key].normalized
        bars = [
            ("Raw", raw_value, group_center - bar_gap / 2 - bar_width, RAW_BAR_COLOR),
            ("Norm", norm_value, group_center + bar_gap / 2, NORM_BAR_COLOR),
        ]
        for label, value, x, color in bars:
            bar_height = (value / ymax) * chart_height
            y = chart_y + chart_height - bar_height
            canvas.rect(x, y, bar_width, bar_height, fill=color, rx=10)
            canvas.text(x + bar_width / 2, y - 10, f"{value * 100:.1f}%", css_class="small", anchor="middle")
            canvas.text(
                x + bar_width / 2,
                chart_y + chart_height + 24,
                label,
                css_class="small muted",
                anchor="middle",
            )

        canvas.text_lines(
            group_center,
            chart_y + chart_height + 54,
            run.label_lines,
            css_class="label",
            anchor="middle",
            line_height=16,
        )

    legend_x = 60
    legend_y = 132
    canvas.rect(legend_x, legend_y, 20, 20, fill=RAW_BAR_COLOR, rx=4)
    canvas.text(legend_x + 30, legend_y + 14, "Raw ranking", css_class="legend")
    canvas.rect(legend_x + 170, legend_y, 20, 20, fill=NORM_BAR_COLOR, rx=4)
    canvas.text(legend_x + 200, legend_y + 14, "Byte-normalized ranking", css_class="legend")

    write_svg(out_path, canvas)


def render_per_class_chart(
    runs: Sequence[RunMetrics],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    values_by_run: str,
) -> None:
    run_count = len(runs)
    width = 1120 + max(0, run_count - 4) * 160
    height = 820
    canvas = SvgCanvas(width, height)
    canvas.text(60, 66, title, css_class="title")
    canvas.text(60, 96, subtitle, css_class="subtitle")

    chart_x = 130
    chart_y = 160
    chart_width = width - chart_x - 90
    chart_height = 410
    values = [per_class_metric_value(run, label, values_by_run) for run in runs for label in ACTIVE_LABELS]
    ymax = axis_max(values, minimum=0.4)
    draw_axis_grid(
        canvas,
        chart_x=chart_x,
        chart_y=chart_y,
        chart_width=chart_width,
        chart_height=chart_height,
        ymax=ymax,
    )

    class_group_width = chart_width / len(ACTIVE_LABELS)
    bar_gap = 12
    bar_width = min(50.0, max(28.0, (class_group_width - 70 - (run_count - 1) * bar_gap) / run_count))
    total_run_width = run_count * bar_width + (run_count - 1) * bar_gap

    for class_index, label in enumerate(ACTIVE_LABELS):
        group_left = chart_x + class_group_width * class_index
        start_x = group_left + (class_group_width - total_run_width) / 2
        for run_index, run in enumerate(runs):
            value = per_class_metric_value(run, label, values_by_run)
            x = start_x + run_index * (bar_width + bar_gap)
            bar_height = (value / ymax) * chart_height
            y = chart_y + chart_height - bar_height
            canvas.rect(x, y, bar_width, bar_height, fill=run_color(run), rx=10)
            canvas.text(x + bar_width / 2, y - 10, f"{value * 100:.1f}%", css_class="small", anchor="middle")

        canvas.text_lines(
            group_left + class_group_width / 2,
            chart_y + chart_height + 36,
            LABEL_LINES[label],
            anchor="middle",
            css_class="label",
            line_height=16,
        )

    llama_runs = [run for run in runs if run.model_family == "llama"]
    gemma_runs = [run for run in runs if run.model_family == "gemma"]
    ordered_legend_runs = llama_runs + gemma_runs + [run for run in runs if run.model_family not in {"llama", "gemma"}]
    legend_columns = max(4, len(llama_runs), len(gemma_runs))
    legend_x_start = 120
    legend_x_gap = (width - 240) / legend_columns
    legend_y_start = 680
    legend_row_gap = 34
    for index, run in enumerate(ordered_legend_runs):
        legend_x = legend_x_start + (index % legend_columns) * legend_x_gap
        legend_y = legend_y_start + (index // legend_columns) * legend_row_gap
        canvas.rect(legend_x, legend_y - 14, 20, 20, fill=run_color(run), rx=4)
        canvas.text(legend_x + 30, legend_y, run.legend_name, css_class="legend")

    write_svg(out_path, canvas)


def render_confusion_chart(run: RunMetrics, out_path: Path) -> None:
    width, height = 860, 780
    canvas = SvgCanvas(width, height)
    canvas.text(60, 66, f"{run.display_name} Confusion Matrix", css_class="title")
    canvas.text(
        60,
        96,
        "Rows are gold labels, columns are predicted labels. Cells show row-normalized percentages.",
        css_class="subtitle",
    )

    grid_left = 260
    grid_top = 220
    cell = 150

    canvas.text(grid_left + 1.5 * cell, 146, "Predicted Label", css_class="label", anchor="middle", weight="700")
    matrix_mid_y = grid_top + 1.5 * cell - 5
    canvas.add(
        '<text transform="translate(92 {y}) rotate(-90)" text-anchor="middle" '
        'class="label" font-weight="700">Gold Label</text>'.format(y=matrix_mid_y)
    )

    for col_index, label in enumerate(ACTIVE_LABELS):
        canvas.text_lines(
            grid_left + col_index * cell + (cell - 10) / 2,
            grid_top - 42,
            LABEL_LINES[label],
            anchor="middle",
            css_class="label",
            line_height=16,
        )

    for row_index, label in enumerate(ACTIVE_LABELS):
        support = run.gold_support[label]
        canvas.text_lines(
            grid_left - 24,
            grid_top + row_index * cell + 44,
            [*LABEL_LINES[label], f"(n={support})"],
            anchor="end",
            css_class="label",
            line_height=15,
        )
        for col_index, count in enumerate(run.normalized_confusion_active[row_index]):
            value = 0.0 if support == 0 else count / support
            fill = lerp_color("#f8fafc", "#1d4ed8", value)
            text_color = "#ffffff" if value >= 0.55 else "#0f172a"
            x = grid_left + col_index * cell
            y = grid_top + row_index * cell
            canvas.rect(x, y, cell - 10, cell - 10, fill=fill, rx=14, stroke="#cbd5e1")
            canvas.text(
                x + (cell - 10) / 2,
                y + 64,
                f"{value * 100:.1f}%",
                anchor="middle",
                weight="700",
                fill=text_color,
            )
            canvas.text(
                x + (cell - 10) / 2,
                y + 92,
                f"{count}/{support}",
                anchor="middle",
                css_class="small",
                fill=text_color,
            )

    canvas.text(
        60,
        734,
        "Only active labels are shown here. The 'direct' class is omitted because it has zero support in these runs.",
        css_class="small muted",
    )

    write_svg(out_path, canvas)


def render_normalization_outcomes(runs: Sequence[RunMetrics], out_path: Path) -> None:
    run_count = len(runs)
    width = 1120
    total_height = (52 + 48) * run_count - 48
    height = 280 + total_height + 140
    canvas = SvgCanvas(width, height)
    canvas.text(60, 66, "What Byte-Length Normalization Changed", css_class="title")
    canvas.text(
        60,
        96,
        "Each bar partitions the test set into examples normalization kept right, fixed, broke, or still missed.",
        css_class="subtitle",
    )

    chart_x = 270
    chart_y = 150
    chart_width = 760
    bar_height = 52
    gap = 48

    total_height = (bar_height + gap) * run_count - gap
    for tick in range(0, 101, 10):
        x = chart_x + chart_width * (tick / 100)
        canvas.line(x, chart_y - 20, x, chart_y + total_height + 16)
        canvas.text(x, chart_y - 28, f"{tick}%", css_class="small muted", anchor="middle")

    legend_labels = {
        "stay_correct": "Stayed correct",
        "fixed": "Fixed by normalization",
        "broken": "Broken by normalization",
        "stay_wrong": "Stayed wrong",
    }

    for index, run in enumerate(runs):
        y = chart_y + index * (bar_height + gap)
        canvas.text_lines(60, y + 20, run.label_lines, css_class="label", weight="700", line_height=16)
        total = len(run.samples)
        segments = [
            ("stay_correct", run.normalization_outcomes["stay_correct"]),
            ("fixed", run.normalization_outcomes["fixed"]),
            ("broken", run.normalization_outcomes["broken"]),
            ("stay_wrong", run.normalization_outcomes["stay_wrong"]),
        ]
        x = chart_x
        for key, count in segments:
            width_px = chart_width * (count / total)
            canvas.rect(x, y, width_px, bar_height, fill=OUTCOME_COLORS[key], rx=12)
            if width_px >= 120:
                canvas.text(
                    x + width_px / 2,
                    y + 22,
                    legend_labels[key],
                    anchor="middle",
                    css_class="small",
                    weight="700",
                    fill="#ffffff",
                )
                canvas.text(
                    x + width_px / 2,
                    y + 40,
                    f"{count / total * 100:.1f}%",
                    anchor="middle",
                    css_class="small",
                    fill="#ffffff",
                )
            x += width_px

        delta = run.aggregate_metrics["accuracy"].normalized - run.aggregate_metrics["accuracy"].raw
        canvas.text(chart_x + chart_width + 18, y + 32, f"{delta * 100:+.1f} pp", weight="700")

    legend_base_y = chart_y + total_height + 52
    legend_positions = [
        ("stay_correct", 60, legend_base_y),
        ("fixed", 320, legend_base_y),
        ("broken", 60, legend_base_y + 32),
        ("stay_wrong", 320, legend_base_y + 32),
    ]
    for key, legend_x, legend_y in legend_positions:
        canvas.rect(legend_x, legend_y - 14, 20, 20, fill=OUTCOME_COLORS[key], rx=4)
        canvas.text(legend_x + 30, legend_y, legend_labels[key], css_class="legend")

    canvas.text(1060, legend_base_y + 32, "Right-side label = net accuracy change", css_class="small muted", anchor="end")
    write_svg(out_path, canvas)


def cleanup_previous_outputs(out_dir: Path, chart_dir: Path) -> None:
    report_path = out_dir / "README.md"
    if report_path.exists():
        report_path.unlink()
    for chart_path in chart_dir.glob("*.svg"):
        chart_path.unlink()


def main() -> None:
    base_dir = repo_root()
    out_dir = ensure_dir(base_dir / "results_report")
    chart_dir = ensure_dir(out_dir / "charts")
    cleanup_previous_outputs(out_dir, chart_dir)

    runs = load_runs(base_dir)

    for metric_key, title, subtitle in METRIC_SPECS:
        render_metric_chart(
            runs,
            chart_dir / f"{metric_key}.svg",
            metric_key=metric_key,
            title=title,
            subtitle=subtitle,
        )

    render_per_class_chart(
        runs,
        chart_dir / "per_class_f1.svg",
        title="Normalized Per-Class F1",
        subtitle="F1 after byte-length normalization for the active labels in the test split.",
        values_by_run="f1",
    )
    render_per_class_chart(
        runs,
        chart_dir / "per_class_accuracy.svg",
        title="Normalized Per-Class Accuracy",
        subtitle="Diagonal of the normalized confusion matrix divided by class support. In this single-label setup, that equals per-class recall.",
        values_by_run="accuracy",
    )
    render_per_class_chart(
        runs,
        chart_dir / "per_class_precision.svg",
        title="Normalized Per-Class Precision",
        subtitle="Precision from the normalized classification report for each active label.",
        values_by_run="precision",
    )
    render_per_class_chart(
        runs,
        chart_dir / "per_class_recall.svg",
        title="Normalized Per-Class Recall",
        subtitle="Recall from the normalized classification report for each active label. In this setup, it matches per-class accuracy.",
        values_by_run="recall",
    )

    for run in runs:
        render_confusion_chart(run, chart_dir / f"confusion_{run.run_key}.svg")

    render_normalization_outcomes(runs, chart_dir / "normalization_outcomes.svg")

    for chart in sorted(chart_dir.glob("*.svg")):
        print(f"Wrote chart {chart}")


if __name__ == "__main__":
    main()
