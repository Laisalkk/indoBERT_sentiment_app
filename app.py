import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from pathlib import Path

# === Path Model di dalam Repo ===
MODEL_DIR = Path(__file__).parent / "model"

# === Load Tokenizer & Model (cache biar cepat) ===
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# === Fungsi Prediksi ===
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    label = "😊 Positif" if pred == 1 else "😞 Negatif"
    conf = probs[0][pred].item()
    return label, conf

# === Tampilan Streamlit ===
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="centered")

st.title("🧠 Analisis Sentimen IndoBERT")
st.markdown("""
Uji kemampuan model **IndoBERT** dalam menganalisis sentimen teks berbahasa Indonesia 🇮🇩  
Masukkan kalimat atau ulasan, lalu lihat apakah hasilnya positif atau negatif!
""")

user_input = st.text_area("📝 Masukkan teks:", "")

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        with st.spinner("Sedang menganalisis..."):
            label, conf = predict_sentiment(user_input)
        st.success(f"**Hasil:** {label}")
        st.info(f"Tingkat keyakinan model: `{conf:.2%}`")
    else:
        st.warning("Masukkan teks terlebih dahulu ya!")

st.markdown("---")
st.caption("🚀 Ditenagai oleh IndoBERT - HuggingFace 🤗")
