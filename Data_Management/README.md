# Data Management

This directory prepares the balanced source dataset used by SFT, CAI, and probe training. The main script samples equal numbers of `tool_call`, `request_for_info`, and `cannot_answer` examples from When2Call splits.

## Main Files

- `build_balanced_sft_dataset.py`: builds `generated_datasets/when2call_balanced_sft.jsonl` and a summary JSON.
- `data_exploration.ipynb`: data exploration notebook.

## Run

```bash
python3 build_balanced_sft_dataset.py --allow-hf-fallback
```

Useful options:

```bash
python3 build_balanced_sft_dataset.py \
  --num-per-class 3000 \
  --dataset-dir local_datasets_train_test \
  --output-jsonl generated_datasets/when2call_balanced_sft.jsonl
```

The generated JSONL is the input for `CAI/run_cai_split.py` and the default training source expected by the probe pipeline.
