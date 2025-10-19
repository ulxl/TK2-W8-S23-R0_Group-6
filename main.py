import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown              # pip install gdown
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
GDRIVE_FILE_ID = "1DL_ShoFOTLtgyyv5-dWgcV8iyM_En3P-"   # <== Ganti dengan ID file Drive Anda
MODEL_NAME     = "fashion_mnist_cnn.h5"

# ---------- FUNGSI UNDUH MODEL ----------
@st.cache_resource(show_spinner=False)
def get_model():
    """Download model (first time only) lalu load ke memory."""
    if not os.path.exists(MODEL_NAME):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading model (first run)..."):
            gdown.download(url, MODEL_NAME, quiet=False)
    try:
        return load_model(MODEL_NAME)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

# ---------- PRE-PROCESSING ----------
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("L")                 # grayscale
    img = img.resize((28, 28))               # resize
    img_array = np.array(img) / 255.0        # normalisasi
    return img_array.reshape(1, 28, 28, 1)

# ---------- LABEL ----------
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ---------- UI ----------
st.set_page_config(page_title="Fashion-MNIST CNN", layout="centered")
st.title("🧥 Klasifikasi Gambar Pakaian (Fashion MNIST)")
st.write("Unggah gambar pakaian untuk diprediksi oleh model CNN.")

# ---------- LOAD MODEL ----------
model = get_model()

# ---------- UPLOAD ----------
uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    st.write("🔍 Memprediksi...")

    # ---------- PREDIKSI ----------
    X = preprocess_image(image)
    prob = model.predict(X)[0]                     # shape (10,)

    # top-3
    idx = np.argsort(prob)[::-1][:3]
    labels = [class_names[i] for i in idx]
    confs  = prob[idx]

    # ---------- TAMPILKAN ----------
    st.success("Top-3 Prediksi:")
    for rank, (lab, c) in enumerate(zip(labels, confs), 1):
        st.write(f"{rank}. **{lab}** – Keyakinan: **{c*100:.2f}%**")