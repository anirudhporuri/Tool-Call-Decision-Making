# Bar-Only Plots (`plot_results_new`)

This folder contains a bar-only plotting pipeline that matches the style of `plot_results` while using the `hmm/` snapshot layout.

## Data Sources

By default, the script reads:

- `hmm/Prompting Evals`
- `hmm/Post-Training Evals`
- `hmm/CAI Evals`
- `hmm/Probe Evals`

It supports either a root directory with those subfolders or a direct run directory.

## Run Naming and Ordering

Runs are displayed with explicit names and grouped order:

1. Prompting
2. Post-Training
3. CAI
4. Probe

This grouping happens inside each model family (Llama, then Gemma).

Example labels:

- `Llama Prompting Zero-shot`
- `Llama Post-Training SFT`
- `Llama CAI DPO (From SFT)`
- `Llama Probe Post-Training DPO (Middle layer)`

For probe runs, only the **single best-performing layer** is kept per run
(selected by highest probe accuracy, with macro-F1 as tiebreaker).

To justify that choice, the pipeline also exports probe-only layer comparisons
for **all** probed layers and records the selected layer + reason in tables/CSVs.

Probe runs sourced from corrupted `Prompting 4-shot` probe evaluations are
treated as pending placeholders and shown as `PENDING`.

## Placeholders (Gemma CAI pending)

If Gemma CAI runs are missing, placeholder rows are injected by default for expected run keys, and affected bars are marked `PENDING`.

Disable placeholders with `--no-placeholders`.

## Generated Plots

- `normalized_accuracy_by_run.pdf`
- `normalized_macro_f1_by_run.pdf`
- `normalized_accuracy_by_run_without_probe.pdf`
- `normalized_macro_f1_by_run_without_probe.pdf`
- `per_class_accuracy_bar_by_run.pdf`
- `per_class_accuracy_bar_by_run_without_probe.pdf`
- `per_class_precision_bar_by_run.pdf`
- `per_class_precision_bar_by_run_without_probe.pdf`
- `per_class_recall_bar_by_run.pdf`
- `per_class_recall_bar_by_run_without_probe.pdf`
- `per_class_f1_bar_by_run.pdf`
- `per_class_f1_bar_by_run_without_probe.pdf`
- `unsupported_direct_prediction_rate_by_run.pdf`
- `unsupported_direct_prediction_rate_by_run_without_probe.pdf`
- `normalization_outcome_breakdown_by_run.pdf`
- `normalization_outcome_breakdown_by_run_without_probe.pdf`
- `normalized_prediction_mix_by_run.pdf`
- `normalized_prediction_mix_by_run_without_probe.pdf`
- `probe_layer_accuracy_comparison.pdf`
- `probe_layer_macro_f1_comparison.pdf`
- `probe_layer_macro_precision_comparison.pdf`
- `probe_layer_macro_recall_comparison.pdf`

No heatmaps are generated.

## Usage

From repo root:

```bash
python3 plot_results_new/analyze_results_bar_only.py
```

Custom output directory:

```bash
python3 plot_results_new/analyze_results_bar_only.py \
  --output-dir plot_results_new/output_custom
```

Custom data directories:

```bash
python3 plot_results_new/analyze_results_bar_only.py \
  --runs-dir hmm \
  --probe-runs-dir "hmm/Probe Evals" \
  --output-dir plot_results_new/output_hmm
```

## Output Structure

- `output/data/*.csv`: aggregated metrics tables
- `output/data/probe_layer_metrics.csv`: all probe layers with ranks and `selection_reason`
- `output/data/probe_best_layer_selection.csv`: one selected `BEST` layer per probe setup
- `output/figures/*.pdf`: bar plots
- `output/tables/*.txt`: LaTeX tables (one per figure, with matching base name)
- `output/tables/raw_vs_normalized_change_table_latex.txt`: legacy-style `+/-` normalization table
- `output/tables/byte_normalization_effect_table_latex.txt`: extended `+/-` normalization impact table
- `output/analysis_summary.md`: quick summary and placeholder list
