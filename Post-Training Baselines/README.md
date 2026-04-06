# When2Call SFT and DPO training scripts (zero-shot prompt style)

These scripts train on the released `train_sft` and `train_pref` splits using the same zero-shot tool-use prompt family as the prompting baselines.

Files:
- `train_sft_lora.py`: LoRA SFT on `train_sft`
- `train_dpo_lora.py`: LoRA DPO on `train_pref`
- `w2c_train_format.py`: zero-shot prompt/response formatting for Llama and Gemma
- `run_sft.sh`, `run_dpo.sh`: local/interactive launchers
- `sft_class.sbatch`, `dpo_class.sbatch`: UMIACS class-account batch scripts

Examples:

```bash
# SFT
bash run_sft.sh meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft

# DPO
bash run_dpo.sh meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo

# Batch submission
sbatch sft_class.sbatch meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft
sbatch dpo_class.sbatch meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo
```

Optional fourth argument to the shell or sbatch scripts: a local JSON file to override the default training split.

Environment variables accepted by the run scripts:
- `HF_TOKEN`
- `LOAD_IN_4BIT=1|0`
- `GRADIENT_CHECKPOINTING=1|0`
- `EPOCHS`
- `LR`
- `TRAIN_BS`
- `EVAL_BS`
- `GRAD_ACCUM`
- `MAX_LENGTH`
- `MAX_PROMPT_LENGTH` (DPO only)
- `VAL_SIZE`
- `MAX_TRAIN_SAMPLES`
- `MAX_EVAL_SAMPLES`
- `ATTN_IMPL`
- `REPORT_TO`
