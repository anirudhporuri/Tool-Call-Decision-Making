# Prompting Baselines

This directory evaluates base and trained models on the When2Call multiple-choice task using zero-shot or few-shot prompts.

## Main Files

- `run_eval.py`: launcher for evaluation.
- `w2c_prompts.py`: prompt renderer.
- `fewshot_examples.template.json`: examples for four-shot prompting.
- `eval_class.slurm`: slurm job if running on cluster

## Run

From this directory:
```bash
python3 run_eval.py meta-llama/Llama-3.2-3B-Instruct llama 0 outputs/llama32_3b_zeroshot
python3 run_eval.py meta-llama/Llama-3.2-3B-Instruct llama 4 outputs/llama32_3b_4shot fewshot_examples.template.json
```

Evaluate a LoRA adapter by passing its output directory as the model:

```bash
python3 run_eval.py ../Post-Training\ Baselines/outputs/llama_sft llama 0 outputs/llama_sft_zeroshot_eval
```

Cluster runs use:
```bash
sbatch eval_class.slurm MODEL llama 0 outputs/RUN_NAME
```
