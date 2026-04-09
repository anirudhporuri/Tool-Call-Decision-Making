# When2Call SFT and DPO training scripts (zero-shot prompt style)

These scripts train on the released `train_sft` and `train_pref` splits using the same zero-shot tool-use prompt family as the prompting baselines.

Files:
- `train_sft_lora.py`: LoRA SFT on `train_sft`
- `train_dpo_lora.py`: LoRA DPO on `train_pref`
- `w2c_train_format.py`: zero-shot prompt/response formatting for Llama and Gemma
- `run_sft.py`, `run_dpo.py`: local/interactive Python launchers
- `run_sft.sh`, `run_dpo.sh`: thin shell wrappers around the Python launchers
- `sft_class.slurm`, `dpo_class.slurm`: UMIACS class-account batch scripts

Examples:

```bash
# SFT
python run_sft.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft

# DPO
python run_dpo.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo

# Dry run: validates dataset loading and prompt formatting without loading model weights
python run_sft.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft_dry --dry-run
python run_dpo.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo_dry --dry-run

# Smoke run: loads the model, but only runs a tiny number of examples/steps
python run_sft.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft_smoke --smoke-run
python run_dpo.py meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo_smoke --smoke-run

# Batch submission
sbatch sft_class.slurm meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_sft
sbatch dpo_class.slurm meta-llama/Llama-3.2-3B-Instruct llama outputs/llama_dpo
```

Optional fourth argument to the Python launcher, shell wrapper, or `.slurm` file: a local JSON file to override the default training split.

The Python launchers expose the main settings as argparse flags. They still respect environment variables as defaults for cluster convenience:
- `HF_TOKEN`
- `DATASET_DIR`
- `DRY_RUN=1|0`
- `SMOKE_RUN=1|0`
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
- `MAX_STEPS`
- `ATTN_IMPL`
- `REPORT_TO`

By default, the post-training launchers now persist hub datasets under `../local_datasets` relative to the repo root. Set `DATASET_DIR=/your/shared/path` on the cluster if you want a different reusable location.

To evaluate a post-trained LoRA adapter with the same zero-shot prompting baseline, point the prompting eval script at the saved adapter directory:

```bash
cd ../Prompting\ Baselines
python run_eval.py ../Post-Training\ Baselines/outputs/llama_sft llama 0 outputs/llama_sft_zeroshot_eval
```

That works because the prompting evaluator now detects `adapter_config.json`, loads the base model named by the adapter, and applies the LoRA weights before scoring.
