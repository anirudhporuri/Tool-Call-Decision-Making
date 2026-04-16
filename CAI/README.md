# Constitutional AI pipeline

This folder contains a local, no-API Constitutional AI workflow for the tool-calling setup in this repo.

The recommended pipeline is now stage-based:

1. Split the balanced SFT source JSONL into balanced CAI source halves
2. Generate initial policy outputs on the SFT split and/or DPO split
3. Generate constitution-guided critiques with a pluggable critic model
4. Generate revisions with a pluggable revision model
5. Build the final CAI-SFT dataset
6. Generate DPO response pairs
7. Judge the DPO pairs and directly build the final CAI-DPO dataset
8. Train CAI-SFT / CAI-DPO models with the existing post-training baseline
9. Evaluate the resulting adapters with the existing 0-shot prompting eval

This makes it easy to mix:
- a student policy model for initial outputs / revisions / pair generation
- a larger critic model for critiques and DPO judging

The CAI generation steps still reuse the repo's zero-shot tool-use prompt family through `w2c_train_format.py`.

## Files

- `Constitution.txt`: full constitution
- `critique_prompt.txt`, `revision_prompt.txt`, `preference_prompt.txt`: CAI prompt templates
- `run_cai_split.py`: balanced 50/50 CAI source split, defaulting to `Data_Management/generated_datasets/when2call_balanced_sft.jsonl`
- `run_cai_initial_outputs.py`: generate one policy output per source row
- `run_cai_critiques.py`: generate critiques for a set of initial outputs
- `run_cai_revisions.py`: generate revised outputs from critiques
- `build_cai_sft_dataset.py`: build the final CAI-SFT dataset from initial outputs + critiques + revisions
- `run_cai_response_pairs.py`: generate DPO response pairs
- `run_cai_preferences.py`: judge response pairs and directly build the final CAI-DPO dataset
- `run_cai_sft_data.py`, `run_cai_dpo_data.py`: older monolithic generators kept for reference
- `*.sh`: thin shell wrappers
- `*.slurm`: class-account batch launchers for data-generation stages

## Local examples

```bash
cd CAI

# 1. Prepare balanced source halves
python3 run_cai_split.py

# 2. Generate initial SFT outputs with the student policy model
python3 run_cai_initial_outputs.py \
  google/gemma-3-4b-it \
  gemma \
  outputs/gemma_sft_initial \
  generated_datasets/train_pref_cai_sft_source.jsonl

# 3. Critique those outputs with a stronger critic model
python3 run_cai_critiques.py \
  Qwen/Qwen2.5-7B-Instruct \
  qwen \
  outputs/gemma_sft_critiques \
  generated_datasets/train_pref_cai_sft_source.jsonl \
  outputs/gemma_sft_initial/initial_outputs.jsonl

# 4. Revise with the student policy model (or swap in another reviser)
python3 run_cai_revisions.py \
  google/gemma-3-4b-it \
  gemma \
  outputs/gemma_sft_revisions \
  generated_datasets/train_pref_cai_sft_source.jsonl \
  outputs/gemma_sft_initial/initial_outputs.jsonl \
  outputs/gemma_sft_critiques/critiques.jsonl

# 5. Build the final CAI-SFT dataset
python3 build_cai_sft_dataset.py \
  outputs/gemma_cai_sft_dataset \
  generated_datasets/train_pref_cai_sft_source.jsonl \
  outputs/gemma_sft_initial/initial_outputs.jsonl \
  outputs/gemma_sft_critiques/critiques.jsonl \
  outputs/gemma_sft_revisions/revisions.jsonl

# 6. Train the CAI-SFT model
cd ../Post-Training\ Baselines
python3 run_sft.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft ../CAI/outputs/gemma_cai_sft_dataset/cai_sft_dataset.jsonl

# 7. Generate DPO response pairs from the base or SFT policy model
cd ../CAI
python3 run_cai_response_pairs.py \
  ../Post-Training\ Baselines/outputs/gemma_cai_sft_self \
  gemma \
  outputs/gemma_dpo_pairs \
  generated_datasets/train_pref_cai_dpo_source.jsonl

# 8. Judge the pairs and directly build the DPO dataset
python3 run_cai_preferences.py \
  Qwen/Qwen2.5-7B-Instruct \
  qwen \
  outputs/gemma_cai_dpo_dataset \
  generated_datasets/train_pref_cai_dpo_source.jsonl \
  outputs/gemma_dpo_pairs/response_pairs.jsonl

# 9. Train CAI-DPO
cd ../Post-Training\ Baselines
python3 run_dpo.py ../Post-Training\ Baselines/outputs/gemma_cai_sft gemma outputs/gemma_cai_dpo ../CAI/outputs/gemma_cai_dpo_dataset/cai_dpo_dataset.jsonl
```

## Colab sketch

```python
%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_split.py

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_initial_outputs.py google/gemma-3-4b-it gemma outputs/gemma_sft_initial generated_datasets/train_pref_cai_sft_source.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_critiques.py Qwen/Qwen2.5-7B-Instruct qwen outputs/gemma_sft_critiques generated_datasets/train_pref_cai_sft_source.jsonl outputs/gemma_sft_initial/initial_outputs.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_revisions.py google/gemma-3-4b-it gemma outputs/gemma_sft_revisions generated_datasets/train_pref_cai_sft_source.jsonl outputs/gemma_sft_initial/initial_outputs.jsonl outputs/gemma_sft_critiques/critiques.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u build_cai_sft_dataset.py outputs/gemma_cai_sft_dataset generated_datasets/train_pref_cai_sft_source.jsonl outputs/gemma_sft_initial/initial_outputs.jsonl outputs/gemma_sft_critiques/critiques.jsonl outputs/gemma_sft_revisions/revisions.jsonl

%cd /content/Tool-Call-Decision-Making/Post-Training\ Baselines
!python3 -u run_sft.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft ../CAI/outputs/gemma_cai_sft_dataset/cai_sft_dataset.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_response_pairs.py /content/Tool-Call-Decision-Making/Post-Training\ Baselines/outputs/gemma_cai_sft gemma outputs/gemma_dpo_pairs generated_datasets/train_pref_cai_dpo_source.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_preferences.py Qwen/Qwen2.5-7B-Instruct qwen outputs/gemma_cai_dpo_dataset generated_datasets/train_pref_cai_dpo_source.jsonl outputs/gemma_dpo_pairs/response_pairs.jsonl

%cd /content/Tool-Call-Decision-Making/Post-Training\ Baselines
!python3 -u run_dpo.py /content/Tool-Call-Decision-Making/Post-Training\ Baselines/outputs/gemma_cai_sft gemma outputs/gemma_cai_dpo ../CAI/outputs/gemma_cai_dpo_dataset/cai_dpo_dataset.jsonl
```

All final evals should use the existing prompting baseline in 0-shot mode.
