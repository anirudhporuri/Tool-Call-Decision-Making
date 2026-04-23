# Bar-Only Plots (`plot_results_new`)

This folder contains a new plotting pipeline that mirrors the visual style of `plot_results/analyze_results.py`, but only generates bar-style plots (no heatmaps).

## What this script does

- Uses the same general look-and-feel:
  - `plotnine`
  - `theme_bw`
  - horizontal bars
  - consistent family/class color palette
- Aggregates prompting eval runs and probe eval runs (if present).
- Writes CSV tables under `output/data`.
- Writes bar plots under `output/figures`.
- Injects **placeholder bars** for missing Gemma CAI eval runs by default.

## Placeholder logic (Gemma CAI pending)

When Gemma CAI eval runs are incomplete, the script infers missing runs from expected naming patterns and inserts placeholder rows.

Expected Gemma CAI eval keys include:

- `gemma_self_full_sft_model_eval`
- `gemma_self_full_dpo_base_model_eval`
- `gemma_self_full_dpo_from_sft_model_eval`

Additionally, if llama CAI eval runs exist, the script mirrors those names into Gemma equivalents and fills missing ones.

Placeholders are rendered as:

- gray bars in run-level plots
- `PENDING` labels on affected runs
- zero-valued placeholder rows in CSV outputs so ordering remains stable

Disable placeholders with `--no-placeholders`.

## Generated plots

- `normalized_accuracy_by_run.pdf`
- `normalized_macro_f1_by_run.pdf`
- `normalized_accuracy_by_run_without_probe.pdf`
- `normalized_macro_f1_by_run_without_probe.pdf`
- `per_class_accuracy_bar_by_run.pdf`
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

With explicit directories (for example, `hmm` snapshots):

```bash
python3 plot_results_new/analyze_results_bar_only.py \
  --runs-dir "hmm/Prompting Evals" \
  --probe-runs-dir "hmm/Probe Evals" \
  --output-dir "plot_results_new/output_hmm"
```

If you want only real runs and no placeholders:

```bash
python3 plot_results_new/analyze_results_bar_only.py --no-placeholders
```

## Output structure

- `output/data/*.csv`: aggregated metrics tables
- `output/figures/*.pdf`: bar plots
- `output/analysis_summary.md`: quick summary + placeholder list
