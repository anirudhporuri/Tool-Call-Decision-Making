# Constitutional AI pipeline

This folder contains a local, no-API Constitutional AI workflow for the tool-calling setup in this repo.

The pipeline is:

1. Split `train_pref` into balanced CAI source halves
2. Generate CAI-SFT data from constitution-guided revisions
3. Train CAI-SFT models with the existing post-training baseline
4. Generate CAI-DPO data from AI preferences
5. Train CAI-DPO models with the existing post-training baseline
6. Evaluate the resulting adapters with the existing 0-shot prompting eval

The CAI generation steps reuse the repo's zero-shot tool-use prompt family through `w2c_train_format.py`.

## Files

- `Constitution.txt`: full constitution
- `critique_prompt.txt`, `revision_prompt.txt`, `preference_prompt.txt`: CAI prompt templates
- `run_cai_split.py`: balanced 50/50 CAI source split from `train_pref`
- `run_cai_sft_data.py`: generate CAI-SFT datasets for one student family
- `run_cai_dpo_data.py`: generate CAI-DPO dataset for one student family and branch
- `*.sh`: thin shell wrappers
- `*.slurm`: class-account batch launchers for data-generation stages

## Local examples

```bash
cd CAI

# 1. Prepare balanced source halves
python3 run_cai_split.py --allow-hf-fallback

# 2. Generate CAI-SFT data for Gemma
python3 run_cai_sft_data.py \
  google/gemma-3-4b-it \
  gemma \
  outputs/gemma_cai_sft_data \
  generated_datasets/train_pref_cai_sft_source.jsonl

# 3. Train CAI-SFT branches
cd ../Post-Training\ Baselines
python3 run_sft.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft_self ../CAI/outputs/gemma_cai_sft_data/cai_sft_gemma_self.jsonl
python3 run_sft.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft_cross ../CAI/outputs/gemma_cai_sft_data/cai_sft_gemma_cross.jsonl

# 4. Generate CAI-DPO data
cd ../CAI
python3 run_cai_dpo_data.py \
  ../Post-Training\ Baselines/outputs/gemma_cai_sft_self \
  gemma \
  self \
  outputs/gemma_cai_dpo_self_data \
  generated_datasets/train_pref_cai_dpo_source.jsonl

# 5. Train CAI-DPO branch
cd ../Post-Training\ Baselines
python3 run_dpo.py ../CAI/outputs/gemma_cai_sft_self gemma outputs/gemma_cai_dpo_self ../CAI/outputs/gemma_cai_dpo_self_data/cai_dpo_gemma_self.jsonl
```

## Cluster examples

```bash
cd CAI
sbatch cai_sft_data_class.slurm google/gemma-3-4b-it gemma outputs/gemma_cai_sft_data generated_datasets/train_pref_cai_sft_source.jsonl

cd CAI
sbatch cai_dpo_data_class.slurm ../Post-Training\ Baselines/outputs/gemma_cai_sft_self gemma self outputs/gemma_cai_dpo_self_data generated_datasets/train_pref_cai_dpo_source.jsonl
```

## Colab sketch

```python
%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_split.py --allow-hf-fallback

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_sft_data.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft_data generated_datasets/train_pref_cai_sft_source.jsonl

%cd /content/Tool-Call-Decision-Making/Post-Training\ Baselines
!python3 -u run_sft.py google/gemma-3-4b-it gemma outputs/gemma_cai_sft_self ../CAI/outputs/gemma_cai_sft_data/cai_sft_gemma_self.jsonl

%cd /content/Tool-Call-Decision-Making/CAI
!python3 -u run_cai_dpo_data.py /content/Tool-Call-Decision-Making/Post-Training\ Baselines/outputs/gemma_cai_sft_self gemma self outputs/gemma_cai_dpo_self_data generated_datasets/train_pref_cai_dpo_source.jsonl

%cd /content/Tool-Call-Decision-Making/Post-Training\ Baselines
!python3 -u run_dpo.py /content/Tool-Call-Decision-Making/Post-Training\ Baselines/outputs/gemma_cai_sft_self gemma outputs/gemma_cai_dpo_self ../CAI/outputs/gemma_cai_dpo_self_data/cai_dpo_gemma_self.jsonl
```

All final evals should use the existing prompting baseline in 0-shot mode.
