import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Oral Homogenous Leukoplakia",
    "Oral Non-Homogenous Leukoplakia",
    "Other Oral White Lesions"
]

st.set_page_config(page_title="OralScan AI", page_icon="🦷")
st.title("🦷 OralScan AI — Analysis Studio")
st.caption("Upload a dental image for AI-powered oral lesion screening")

@st.cache_resource
def load_model():
    import keras
    return keras.saving.load_model('./model.keras')

model = load_model()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analysing..."):
        arr = (np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 127.5) - 1.0
        preds = model.predict(np.expand_dims(arr, 0))[0]
    idx = int(np.argmax(preds))
    st.subheader("Results")
    st.success(f"**{CLASS_NAMES[idx]}** — {round(float(preds[idx])*100, 1)}% confidence")
    for i, name in enumerate(CLASS_NAMES):
        st.progress(float(preds[i]), text=f"{name}: {round(float(preds[i])*100,1)}%")
    st.warning("⚠️ For screening purposes only. Consult a qualified clinician.")
