# Constitutional AI Pipeline

This directory now supports the staged Constitutional AI workflow only. The older monolithic CAI data generators have been removed so the repo matches the pipeline we actually run on cluster.

The staged flow is:

1. Split the balanced source set into CAI SFT and CAI DPO halves
2. Generate initial outputs
3. Generate critiques
4. Generate revisions
5. Build the CAI SFT dataset
6. Train and evaluate the CAI SFT adapter
7. Generate DPO response pairs
8. Judge those pairs and build the CAI DPO dataset
9. Train and evaluate DPO adapters from the base model and from the SFT adapter

`cai_full_pipeline_common.py` already supports resume-friendly reruns through `--skip-completed`, which is on by default. If a batch job dies mid-run, rerunning the same wrapper will skip stages whose expected artifacts are already complete.

## Supported Full-Pipeline Modes

The preferred full-pipeline entrypoints are:

- `cai_full_pipeline_gemma_self_class.slurm`
- `cai_full_pipeline_gemma_qwen_class.slurm`
- `cai_full_pipeline_llama_self_class.slurm`
- `cai_full_pipeline_llama_qwen_class.slurm`

These correspond to:

- Gemma self-judge
- Gemma qwen-judge
- Llama self-judge
- Llama qwen-judge

The generic wrappers remain available:

- `cai_full_pipeline_gemma_class.slurm`
- `cai_full_pipeline_llama_class.slurm`

They currently map to the qwen-judge configuration.

## Precision Policy

The full pipeline now routes precision separately for three parts of the run:

- base-generation stages
  - `run_cai_initial_outputs.py`
  - `run_cai_revisions.py`
  - `run_cai_response_pairs.py`
- critic/judge stages
  - `run_cai_critiques.py`
  - `run_cai_preferences.py`
- training stages
  - `run_sft.py`
  - `run_dpo.py`

Current defaults by wrapper:

- self-judge wrappers
  - base-generation: normal precision
  - critic/judge: normal precision
  - training: 4-bit
- qwen-judge wrappers
  - base-generation: 4-bit
  - critic/judge: 4-bit
  - training: 4-bit

This keeps self-judge runs closer to the original model-loading setup while preserving the memory-saving post-training path we already use for SFT and DPO.

## Model Locations

Gemma base model:

- `/fs/class-projects/spring2026/cmsc848q/mukunds/google__gemma-3-4b-it`

Llama base model:

- `meta-llama/Llama-3.2-3B-Instruct`

Qwen critic/judge:

- `Qwen/Qwen3.5-9B`

Qwen is supported here only through the existing Transformers + bitsandbytes runtime 4-bit path. GGUF / Q4_K_M integration is not part of this pipeline refresh.

## Key Files

- `run_cai_split.py`: balanced CAI source split
- `run_cai_initial_outputs.py`: one initial policy output per source row
- `run_cai_critiques.py`: constitution-guided critiques
- `run_cai_revisions.py`: revised outputs from critiques
- `build_cai_sft_dataset.py`: final CAI SFT dataset builder
- `run_cai_response_pairs.py`: DPO pair generation
- `run_cai_preferences.py`: DPO pair judging and final dataset export
- `cai_full_pipeline_common.py`: staged cluster driver with skip-completed behavior

## Typical Cluster Runs

From the `CAI/` directory:

```bash
sbatch cai_full_pipeline_gemma_self_class.slurm
sbatch cai_full_pipeline_gemma_qwen_class.slurm
HF_TOKEN="..." sbatch cai_full_pipeline_llama_self_class.slurm
HF_TOKEN="..." sbatch cai_full_pipeline_llama_qwen_class.slurm
```

To force a stage to rerun instead of resuming:

```bash
sbatch cai_full_pipeline_gemma_self_class.slurm --no-skip-completed
```

All final evaluation still goes through the prompting-baseline evaluation path in 0-shot mode.
