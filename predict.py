import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict(text: str, tokenizer, model, max_length: int):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    logits = model(**tokens).logits
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred = int(probs.argmax())
    label = "Human" if pred == 0 else "Modified/Machine"

    return label, float(probs[0]), float(probs[1])


def run_both(text: str, label: str):
    print(f"\n--- {label} ---")
    print(f"Text preview: {text[:80]}...")

    for name, tokenizer, model, max_len in MODELS:
        lbl, p_human, p_machine = predict(text, tokenizer, model, max_len)
        print(f"[{name}] Prediction: {lbl} | P(Human): {p_human:.4f} | P(Machine): {p_machine:.4f}")


if __name__ == "__main__":
    print("Loading models...")
    roberta_tokenizer, roberta_model = load_model("results_roberta_pan24")
    modernbert_tokenizer, modernbert_model = load_model("results_modernbert_pan24")

    MODELS = [
        ("RoBERTa-base",   roberta_tokenizer,    roberta_model,    128),
        ("ModernBERT-base", modernbert_tokenizer, modernbert_model, 512),
    ]

    machine_text = (
        "Pandemic's Lasting Impact: Millions of Jobs Unlikely to Return, Spurring Career Changes and Retraining "
        "As the world grapples with the ongoing COVID-19 pandemic, a harsh reality is setting in: millions of jobs "
        "affected by the crisis are unlikely to return, necessitating career changes and retraining for the unemployed. "
        "According to a recent report by the McKinsey Global Institute, up to 20% of the workforce in some countries "
        "may need to switch occupational categories or industries to adapt to the new economic landscape."
    )

    human_text = (
        "'Rust' assistant director who handed Alec Baldwin prop gun subpoenaed after declining investigation interview"
        "SANTA FE, N.M. — The assistant director who handed Alec Baldwin a prop gun that killed a cinematographer on a "
        "New Mexico film set must make himself available for an interview with state workplace safety regulators, a judge "
        "has decided. District Judge Bryan Biedscheid on Friday granted a request by the Occupational Health and Safety "
        "Bureau of the state Environment Department to issue a subpoena to Dave Halls, assistant director for the movie "
        "\"Rust,\" local news outlets reported."
    )

    run_both(machine_text, "Modified/Machine text")
    run_both(human_text,   "Human text")
