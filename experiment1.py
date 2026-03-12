import os
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
    set_seed,
)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "precision_pos": p,
        "recall_pos": r,
        "f1_pos": f1,
        "f1_macro": f1_macro,
    }


class WeightedTrainer(Trainer):
    """
    Trainer with class-weighted cross entropy to handle imbalance (your mode=all case).
    """
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(42)

    data_files = {
        "train": "dataset/train.csv",
        "validation": "dataset/dev.csv",
        "test": "dataset/test.csv",
    }

    ds = load_dataset("csv", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Tokenize
    tokenized = ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    # Rename label column
    tokenized = tokenized.rename_column("generated", "labels")

    # Keep only model inputs
    keep_cols = {"input_ids", "attention_mask", "labels"}
    for split in tokenized.keys():
        drop_cols = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(drop_cols)

    tokenized.set_format("torch")

    # Compute class weights (important for your 13:1 imbalance)
    y = tokenized["train"]["labels"].numpy()
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = torch.tensor(weights, dtype=torch.float)

    print("Class weights:", {0: float(class_weights[0]), 1: float(class_weights[1])})

    # W&B only if enabled by env
    report_to = []
    if os.getenv("WANDB_PROJECT"):
        report_to = ["wandb"]

    args = TrainingArguments(
        output_dir="results_deberta_pan24",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,

        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        logging_steps=50,
        save_total_limit=2,

        fp16=torch.cuda.is_available(),  # если есть GPU — ускорит
        report_to=report_to,
        run_name="deberta-pan24-news" if report_to else None,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    print("\nValidation metrics:")
    print(trainer.evaluate())

    print("\nTest metrics:")
    print(trainer.evaluate(tokenized["test"]))

    # Save
    model.save_pretrained("results_deberta_pan24")
    tokenizer.save_pretrained("results_deberta_pan24")
    print("\nSaved model + tokenizer to: results_deberta_pan24")


if __name__ == "__main__":
    main()
