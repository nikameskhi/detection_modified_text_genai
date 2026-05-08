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

## Experiment 3 — ELECTRA-base

**Script:** `experiment3.py`  
**Model:** `google/electra-base-discriminator`

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
| Accuracy | 98.64% | 98.82% |
| F1 macro | 94.38% | 95.20% |
| F1 (AI class) | 99.27% | 99.37% |
| Eval loss | 0.3366 | 0.3295 |

---

## Comparison

| Model | Accuracy (val) | F1 macro (val) | F1 AI class (val) | Eval loss |
|---|---|---|---|---|
| RoBERTa-base | 97.41% | 89.44% | 98.61% | 0.3636 |
| ELECTRA-base | 98.64% | 94.38% | 99.27% | 0.3366 |
| ModernBERT-base | **99.65%** | **98.66%** | **99.81%** | **0.2998** |

ModernBERT-base achieves the best results across all metrics, benefiting from its modern architecture (rotary positional embeddings, Flash Attention 2, longer context window). ELECTRA-base, pre-trained with replaced token detection, outperforms RoBERTa-base despite identical model size. RoBERTa-base was trained with a shorter max sequence length (128 vs 512), which likely contributes to the performance gap.

---

## Web App

An interactive Streamlit app for running inference on any text.

**Features:**
- Select from all trained models via a dropdown (only models whose result folders are present are shown)
- Paste any news article and click **Analyze**
- Shows prediction label, per-class confidence percentages, and progress bars

**Setup:**
```bash
pip install streamlit transformers torch
```

**Run:**
```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`.

---