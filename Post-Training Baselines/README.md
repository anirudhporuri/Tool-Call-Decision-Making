# Post-Training Baselines

This directory trains LoRA SFT and DPO baselines on When2Call data. The trained adapters can then be evaluated with `Prompting Baselines/run_eval.py`.

## Main Files

- `run_sft.py`, `run_dpo.py`: local launchers.
- `train_sft_lora.py`, `train_dpo_lora.py`: training implementations.
- `w2c_train_format.py`: prompt and response formatting.
- `sft_class.slurm`, `dpo_class.slurm`: cluster launchers.

## Run

From this directory:
```bash
python3 run_sft.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft
python3 run_dpo.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo
```

To train from a local JSONL instead of the Hugging Face split, pass it as the fourth argument:
```bash
python run_sft.py MODEL llama outputs/cai_sft ../CAI/outputs/RUN/sft_dataset/cai_sft_dataset.jsonl
```

Cluster runs use the same positional arguments:
```bash
sbatch sft_class.slurm MODEL llama outputs/llama_sft
sbatch dpo_class.slurm MODEL llama outputs/llama_dpo
```
