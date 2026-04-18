# Full Analysis of Prompting + Probe Results

_Generated: 2026-04-17 20:37:47_

## What Normalization Means

- `raw` metrics score model outputs exactly as emitted.
- `normalized` metrics apply your output normalization before scoring (mapping surface-form variants into the target label set).
- Positive `normalized - raw` means normalization fixed output-format mismatches; negative means normalization hurt that run.

## What `75pct` Means

- `75pct` means the probe reads hidden states from roughly 75% through the transformer depth.
- It is an intermediate representation between the middle layer and the final layer.

## Top Runs by Normalized Accuracy

| Rank | Run | Type | Norm Acc | Norm Macro-F1 |
|---|---|---|---:|---:|
| 1 | Llama SFT | Prompting | 79.9% | 59.0% |
| 2 | Llama 4-shot | Prompting | 59.0% | 43.2% |
| 3 | Llama DPO | Prompting | 58.5% | 41.8% |
| 4 | Llama Probe Zero-shot (Middle layer) | Probe | 54.7% | 55.0% |
| 5 | Llama Probe Zero-shot (75% depth layer) | Probe | 52.1% | 52.7% |
| 6 | Gemma Probe Zero-shot (75% depth layer) | Probe | 48.7% | 48.3% |

## Top Runs by Normalized Macro-F1

| Rank | Run | Type | Norm Macro-F1 | Norm Acc |
|---|---|---|---:|---:|
| 1 | Llama SFT | Prompting | 59.0% | 79.9% |
| 2 | Llama Probe Zero-shot (Middle layer) | Probe | 55.0% | 54.7% |
| 3 | Llama Probe Zero-shot (75% depth layer) | Probe | 52.7% | 52.1% |
| 4 | Gemma Probe Zero-shot (Middle layer) | Probe | 48.3% | 48.6% |
| 5 | Gemma Probe Zero-shot (75% depth layer) | Probe | 48.3% | 48.7% |
| 6 | Llama Probe Zero-shot (Last layer) | Probe | 46.4% | 48.4% |

## Normalization Impact (Prompting Runs Only)

- Average accuracy delta (`normalized - raw`): **+0.044**.
- Average macro-F1 delta (`normalized - raw`): **+0.060**.
- Largest accuracy improvement: **Llama DPO** (+0.153).
- Largest macro-F1 improvement: **Llama DPO** (+0.173).
- Most negative normalization impact on accuracy: **Gemma SFT** (-0.046).

## Probe-Layer Performance

| Run | Norm Acc | Norm Macro-F1 |
|---|---:|---:|
| Gemma Probe Zero-shot (Middle layer) | 48.6% | 48.3% |
| Gemma Probe Zero-shot (75% depth layer) | 48.7% | 48.3% |
| Gemma Probe Zero-shot (Last layer) | 46.2% | 45.6% |
| Llama Probe Zero-shot (Middle layer) | 54.7% | 55.0% |
| Llama Probe Zero-shot (75% depth layer) | 52.1% | 52.7% |
| Llama Probe Zero-shot (Last layer) | 48.4% | 46.4% |

## Class-Level Pattern (Normalized)

- Hardest class by average recall: **Request for info** (33.2%).
- Easiest class by average recall: **Tool call** (58.4%).

| Class | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 |
|---|---:|---:|---:|---:|
| Cannot answer | 71.6% | 64.6% | 46.4% | 52.4% |
| Request for info | 67.1% | 43.5% | 33.2% | 35.7% |
| Tool call | 64.7% | 49.5% | 58.4% | 52.5% |

## Unsupported `direct` Prediction Behavior

- Highest normalized `direct` rate: **Gemma SFT** (29.3%).
- Lowest normalized `direct` rate: **Llama SFT** (0.5%).

## Normalization Outcomes (Averaged Over Prompting Runs)

- Average `fixed` fraction: **8.8%**.
- Average `broken` fraction: **4.4%**.
- Net normalization effect: **helpful on average**.

## Bottom Line

- Probe-layer runs are now included directly in the same comparison framework as prompting runs.
- `75pct` means the probe at roughly three-quarters of transformer depth, not a dataset percentage.
- Use the `_without_probe` plots for apples-to-apples prompting-only comparisons.