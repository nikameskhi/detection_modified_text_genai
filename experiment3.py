import re
import numpy as np
import torch
from torch import nn

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

set_seed(42)

MODEL_NAME = "google/electra-base-discriminator"
MAX_LENGTH = 512
MIN_TEXT_LENGTH = 10


# ----------------------------
# TEXT CLEANING
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess(example):
    text = clean_text(example["text"])
    return {"text": text, "generated": example["generated"]}


# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "precision_pos": p, "recall_pos": r, "f1_pos": f1, "f1_macro": f1_macro}


# ----------------------------
# CUSTOM TRAINER
# ----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.loss_fct = None
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_fct is None:
            weights = self.class_weights.to(logits.device).to(logits.dtype) if self.class_weights is not None else None
            self.loss_fct = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)

        labels = labels.to(logits.device)

        if torch.isnan(logits).any() or torch.isnan(labels).any():
            print("NaN detected in logits or labels — skipping batch")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

        loss = self.loss_fct(logits, labels)

        if torch.isnan(loss):
            print("NaN detected in loss — skipping batch")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


# ----------------------------
# MAIN
# ----------------------------
def main():
    data_files = {
        "train": "dataset/train.csv",
        "validation": "dataset/dev.csv",
        "test": "dataset/test.csv",
    }

    ds = load_dataset("csv", data_files=data_files)

    ds = ds.map(preprocess)
    ds = ds.filter(lambda x: len(x["text"].strip()) >= MIN_TEXT_LENGTH)

    print("Dataset sizes after filtering:")
    for split in ds:
        print(split, len(ds[split]))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        text = [t if len(t.strip()) > 0 else tokenizer.pad_token for t in examples["text"]]
        tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
        safe_ids = [[min(t, tokenizer.vocab_size - 1) for t in seq] for seq in tokens["input_ids"]]
        tokens["input_ids"] = safe_ids
        return tokens

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    tokenized = ds.map(tokenize_function, batched=True, num_proc=4)
    tokenized = tokenized.rename_column("generated", "labels")

    keep_cols = {"input_ids", "attention_mask", "labels"}
    for split in tokenized:
        drop_cols = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(drop_cols)

    tokenized.set_format("torch")

    # ----------------------------
    # CLASS WEIGHTS
    # ----------------------------
    y = np.array(tokenized["train"]["labels"])
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    weights = np.clip(weights, 0.5, 1.0)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print("Class weights (clipped):", weights)

    # ----------------------------
    # SAFE DATA COLLATOR
    # ----------------------------
    def safe_collate(batch, tokenizer=tokenizer):
        batch = [b for b in batch if b['attention_mask'].sum() > 0]
        return DataCollatorWithPadding(tokenizer)(batch)

    # ----------------------------
    # TRAINING ARGS
    # ----------------------------
    args = TrainingArguments(
        output_dir="results_electra_pan24",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        logging_steps=50,
        save_total_limit=2,
        bf16=False,
        fp16=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        data_collator=safe_collate,
        class_weights=class_weights,
        label_smoothing=0.1,
    )

    trainer.train()

    print("\nValidation metrics:")
    print(trainer.evaluate())
    print("\nTest metrics:")
    print(trainer.evaluate(tokenized["test"]))

    model.save_pretrained("results_electra_pan24")
    tokenizer.save_pretrained("results_electra_pan24")


if __name__ == "__main__":
    main()
