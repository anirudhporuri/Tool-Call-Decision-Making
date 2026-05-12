Improving Tool Call Decision Making

This repository contains the code and results for evaluating when language models should call tools, ask for more information, or refuse because the requested action is unsupported.

## Directory Map

- `Data_Management/`: builds the balanced training JSONL used by later stages.
- `Prompting Baselines/`: runs zero-shot and few-shot multiple-choice evaluations.
- `Post-Training Baselines/`: trains LoRA SFT and DPO baselines on When2Call data.
- `CAI/`: generates constitutional critiques, revisions, preferences, and CAI training datasets.
- `Probe Baselines/`: trains hidden-state probes and evaluates probe predictions against model behavior.
- `Raw Results/`: saved evaluation outputs used for analysis and plotting.
- `Plot Results/`: plotting script and generated figures from the raw results.

## Requirements

Create a Python env (3.10) and run:

```bash 
pip3 install -r requirements.txt
```

Full model runs need GPU access, and gated Hugging Face models require `HF_TOKEN`.

## Reproduction Order

1. Build the balanced source data:

   ```bash
   python3 Data_Management/build_balanced_sft_dataset.py --allow-hf-fallback
   ```
   
2. Run prompting baselines from `Prompting Baselines/`.
3. Train SFT/DPO baselines from `Post-Training Baselines/`, then evaluate them with the prompting evaluator.
4. Run CAI pipelines from `CAI/`.
5. Train probes from `Probe Baselines/` using saved evaluation samples.

Each main code directory has its own short README with the exact commands for that stage.