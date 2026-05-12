# CAI

This directory runs the constitutional AI stages used to create CAI SFT and DPO training data. It starts from the balanced JSONL built in `Data_Management/`, generates model outputs, critiques them against `Constitution.txt`, revises them, and judges response pairs.

## Main Files

- `run_cai_split.py`: splits the balanced source JSONL into CAI SFT and DPO sources.
- `run_cai_initial_outputs.py`, `run_cai_critiques.py`, `run_cai_revisions.py`: staged CAI SFT data generation.
- `build_cai_sft_dataset.py`: exports the final CAI SFT JSONL.
- `run_cai_response_pairs.py`, `run_cai_preferences.py`: builds DPO preference data.
- `cai_full_pipeline_common.py`: orchestrates the full CAI, training, and eval pipeline.
- `cai_full_pipeline_*.slurm`: cluster entry points for the main runs.

## Run

From the repo root, first build `Data_Management/generated_datasets/when2call_balanced_sft.jsonl`.

For the full cluster pipeline, use one of:

```bash
sbatch cai_full_pipeline_llama_self_class.slurm
sbatch cai_full_pipeline_gemma_self_class.slurm
```

For a local staged run:

```bash
python3 run_cai_split.py --source-jsonl Data_Management/generated_datasets/when2call_balanced_sft.jsonl
python3 run_cai_initial_outputs.py MODEL llama CAI/outputs/RUN/sft_initial CAI/generated_datasets/train_pref_cai_sft_source.jsonl
python3 run_cai_critiques.py CRITIC_MODEL qwen CAI/outputs/RUN/sft_critiques CAI/generated_datasets/train_pref_cai_sft_source.jsonl CAI/outputs/RUN/sft_initial/initial_outputs.jsonl
python3 run_cai_revisions.py MODEL llama CAI/outputs/RUN/sft_revisions CAI/generated_datasets/train_pref_cai_sft_source.jsonl CAI/outputs/RUN/sft_initial/initial_outputs.jsonl CAI/outputs/RUN/sft_critiques/critiques.jsonl
python3 build_cai_sft_dataset.py CAI/outputs/RUN/sft_dataset CAI/generated_datasets/train_pref_cai_sft_source.jsonl CAI/outputs/RUN/sft_initial/initial_outputs.jsonl CAI/outputs/RUN/sft_critiques/critiques.jsonl CAI/outputs/RUN/sft_revisions/revisions.jsonl
```

For DPO data, use `run_cai_response_pairs.py` on `train_pref_cai_dpo_source.jsonl`, then pass its `response_pairs.jsonl` to `run_cai_preferences.py`.

Use `--dry-run` or `--smoke-run` on scripts before launching full jobs.
