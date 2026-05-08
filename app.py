import os
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    "RoBERTa-base": {
        "dir": "results_roberta_pan24",
        "max_length": 128,
    },
    "ModernBERT-base": {
        "dir": "results_modernbert_pan24",
        "max_length": 512,
    },
    "ELECTRA-base": {
        "dir": "results_electra_pan24",
        "max_length": 512,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict(text, tokenizer, model, max_length):
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
    label = "Human" if pred == 0 else "Modified / Machine-generated"
    return label, float(probs[0]), float(probs[1])


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI Text Detector", page_icon="", layout="centered")

st.title("AI-Modified Text Detector")
st.caption("Detects whether a news article is human-written or AI-generated/modified.")

st.divider()

available = {name: cfg for name, cfg in MODELS.items() if os.path.isdir(cfg["dir"])}

if not available:
    st.error("No trained models found. Make sure at least one results folder exists.")
    st.stop()

model_name = st.selectbox("Select model", list(available.keys()))

text_input = st.text_area(
    "Paste your news article text here",
    height=250,
    placeholder="Enter text to analyze...",
)

predict_btn = st.button("Analyze", type="primary", use_container_width=True)

if predict_btn:
    if not text_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        cfg = available[model_name]
        with st.spinner(f"Loading {model_name}..."):
            tokenizer, model = load_model(cfg["dir"])

        with st.spinner("Analyzing..."):
            label, p_human, p_machine = predict(text_input, tokenizer, model, cfg["max_length"])

        st.divider()

        is_human = label == "Human"
        if is_human:
            st.success(f"**{label}**")
        else:
            st.error(f"**{label}**")

        st.subheader("Confidence")
        col1, col2 = st.columns(2)
        col1.metric("Human", f"{p_human * 100:.2f}%")
        col2.metric("Modified / Machine", f"{p_machine * 100:.2f}%")

        st.progress(p_human, text="Human")
        st.progress(p_machine, text="Modified / Machine")

        st.divider()
        st.caption(f"Model: {model_name} · Device: {str(device).upper()}")
