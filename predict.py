import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "results_deberta_pan24"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


@torch.inference_mode()
def predict(text: str):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    logits = model(**tokens).logits
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred = int(probs.argmax())
    label = "Human" if pred == 0 else "Modified/Machine"
    return label, float(probs[0]), float(probs[1])


if __name__ == "__main__":
    sample_text = "Authorities have unveiled a series of new initiatives aimed at addressing rising inflation..."
    label, p_human, p_machine = predict(sample_text)
    print("Prediction:", label)
    print("P(Human):", p_human)
    print("P(Machine):", p_machine)
