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

## Placeholders (Gemma CAI pending)

If Gemma CAI runs are missing, placeholder rows are injected by default for expected run keys, and affected bars are marked `PENDING`.

Disable placeholders with `--no-placeholders`.

## Generated Plots

- `normalized_accuracy_by_run.pdf`
- `normalized_macro_f1_by_run.pdf`
- `normalized_accuracy_by_run_without_probe.pdf`
- `normalized_macro_f1_by_run_without_probe.pdf`
- `per_class_accuracy_bar_by_run.pdf`
- `per_class_precision_bar_by_run.pdf`
- `per_class_recall_bar_by_run.pdf`
- `per_class_f1_bar_by_run.pdf`
- `unsupported_direct_prediction_rate_by_run.pdf`
- `normalization_outcome_breakdown_by_run.pdf`
- `normalized_prediction_mix_by_run.pdf`
- `normalized_prediction_mix_by_run_without_probe.pdf`

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
- `output/figures/*.pdf`: bar plots
- `output/analysis_summary.md`: quick summary and placeholder list
