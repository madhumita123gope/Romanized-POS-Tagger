import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("model.pkl")
word_enc = joblib.load("word_enc.pkl")
tag_enc = joblib.load("tag_enc.pkl")

def predict_pos(sentence):
    words = sentence.strip().split()
    output = []

    for word in words:
        try:
            encoded = word_enc.transform([word])
            pred = model.predict(np.array(encoded).reshape(-1, 1))
            tag = tag_enc.inverse_transform(pred)[0]
        except ValueError:
            tag = "UNK"  # Unknown word
        output.append((word, tag))

    return output

# Streamlit App UI
st.set_page_config(page_title="POS Tagger", page_icon="🔤", layout="centered")

st.markdown("<h1 style='text-align: center;'>🔠 Romanized POS Tagger</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type a sentence below like a search box 👇</p>", unsafe_allow_html=True)

sentence = st.text_input("", placeholder="e.g. ami tula prem karto", key="input")

if sentence:
    st.markdown("---")
    st.markdown("### 🧾 Tagged Output")
    for word, tag in predict_pos(sentence):
        st.markdown(f"- **{word}** → `{tag}`")
