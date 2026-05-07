# Detection of Modified Text Using GenAI

Binary classification task to detect whether a news article is human-written or AI-generated/modified.

## Dataset

**PAN24 Generative Authorship — News**

| Split | Samples |
|---|---|
| Train | 206,771 |
| Validation | 43,939 |
| Test | 45,570 |

Each sample contains a news article text and a binary label: `0` = human, `1` = AI-generated/modified.

---

## Experiment 1 — RoBERTa-base

**Script:** `experiment1.py`  
**Model:** `roberta-base`

### Setup

| Parameter | Value |
|---|---|
| Max sequence length | 128 |
| Learning rate | 1e-6 |
| Epochs | 3 |
| Batch size (effective) | 8 |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Precision | FP32 |

### Results (Validation)

| Metric | Score |
|---|---|
| Accuracy | 97.41% |
| F1 macro | 89.44% |
| F1 (AI class) | 98.61% |
| Eval loss | 0.3636 |

---

## Experiment 2 — ModernBERT-base

**Script:** `experiment2.py`  
**Model:** `answerdotai/ModernBERT-base`

### Setup

| Parameter | Value |
|---|---|
| Max sequence length | 512 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size (effective) | 16 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Precision | FP16 |

### Results

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 99.65% | 99.56% |
| F1 macro | 98.66% | 98.35% |
| F1 (AI class) | 99.81% | 99.77% |
| Eval loss | 0.2998 | 0.3019 |

---