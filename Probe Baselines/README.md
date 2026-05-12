# Probe Baselines

This directory trains logistic probes on hidden states and compares probe predictions with final model behavior. 

## Main Files

- `run_probe.py`: launcher for probe jobs.
- `w2c_probe.py`: hidden-state extraction, probe training, and evaluation.
- `probe_*.slurm`: slurm jobs

## Run

From this directory:
```bash
python3 run_probe.py MODEL llama outputs/llama_probe ../Raw\ Results/Prompting\ Evals/llama32_3b_zeroshot/samples.jsonl
```

For four-shot probing:
```bash
python3 run_probe.py MODEL llama outputs/llama_probe_4shot ../Raw\ Results/Prompting\ Evals/llama32_3b_4shot/samples.jsonl --use-4shot-prompt
```

For adapter checkpoints, pass the adapter path as `MODEL` and optionally set `--peft-base-model-override`.

Cluster examples:
```bash
sbatch probe_llama.slurm
sbatch probe_gemma.slurm
```

Important outputs are `probe_evaluation_summary.json`, `probe_training_summary.json`, `probe_comparison_samples.jsonl`, and `probe_model.pkl`.
