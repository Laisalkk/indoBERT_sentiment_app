import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import os, re

# ==============================
# Konfigurasi Model
# ==============================
MODEL_NAME = "laisalkk/indoBERT-caption"

@st.cache_resource
def load_model():
    # Login ke Hugging Face pakai token dari Streamlit Secret
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==============================
# Fungsi bantu pembersihan teks
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?()/-]", "", text)
    return text.strip()

def normalize_capitalization(text):
    if not text:
        return ""
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text

# ==============================
# Fungsi Generate Caption
# ==============================
def generate_captions(tokenizer, model, label, title, isi, num_captions=2):
    isi_clean = clean_text(isi)
    title_clean = clean_text(title)
    prompt = f"Label: {label}. Judul: {title_clean}. Isi: {isi_clean}."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    captions = []
    for _ in range(num_captions):
        output_ids = model.generate(
            **inputs,
            max_length=380,
            num_beams=5,
            length_penalty=1.5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.9,
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(normalize_capitalization(caption))
    return captions

# ==============================
# UI Streamlit
# ==============================
st.set_page_config(page_title="IndoBERT Caption Generator", page_icon="🧠", layout="centered")
st.title("🧠 IndoBERT Caption Generator")
st.caption("Model: laisalkk/indoBERT-caption")

judul = st.text_input("📰 Judul Berita:")
isi = st.text_area("📄 Isi Berita:")
label = st.selectbox("🏷️ Label Berita:", ["Fakta", "Hoaks"])
num_captions = st.slider("🔢 Jumlah Caption yang ingin dihasilkan:", 1, 3, 2)

if st.button("🚀 Generate Caption"):
    if not isi.strip():
        st.warning("Masukkan isi berita terlebih dahulu.")
    else:
        with st.spinner("Model sedang menghasilkan caption..."):
            captions = generate_captions(tokenizer, model, label, judul, isi, num_captions)

        st.success("✅ Caption berhasil dibuat!")
        for i, cap in enumerate(captions, start=1):
            st.markdown(f"**🟢 Caption {i}:** {cap}")

st.markdown("---")
st.caption("Ditenagai oleh IndoBERT2BERT fine-tuned 🇮🇩 oleh @laisalkk")
