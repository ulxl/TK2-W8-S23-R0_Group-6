import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ---------- PRE-PROCESS ----------
def preprocess_image(image):
    img = image.convert('L')          # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 1)

# ---------- LABEL ----------
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ---------- UI ----------
st.title("Klasifikasi Gambar Pakaian (Fashion MNIST)")
st.write("Unggah gambar pakaian untuk diprediksi oleh model CNN.")

model = load_model('fashion_mnist_cnn.h5')

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_container_width=True)

    # ---------- PREDIKSI + TOP-3 ----------
    with st.spinner("Memprediksi..."):
        processed_image = preprocess_image(image)
        prob = model.predict(processed_image)[0]        # shape (10,)

    # ambil 3 besar
    top3_idx = np.argsort(prob)[::-1][:3]
    top3_label = [class_names[i] for i in top3_idx]
    top3_prob  = prob[top3_idx]

    # ---------- TAMPILKAN HANYA TOP-3 ----------
    st.success("Top-3 Prediksi:")
    for rank, (label, confidence) in enumerate(zip(top3_label, top3_prob), 1):
        st.write(f"{rank}. **{label}** – Keyakinan: **{confidence*100:.2f}%**")