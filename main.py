import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ---------- Fungsi Pre-processing ----------
def preprocess_image(image):
    img = image.convert('L')          # Grayscale
    img = img.resize((28, 28))        # Resize
    img_array = np.array(img) / 255.0 # Normalisasi
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# ---------- Label Fashion-MNIST ----------
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ---------- UI Streamlit ----------
st.title("Klasifikasi Gambar Pakaian (Fashion MNIST)")
st.write("Unggah gambar pakaian untuk diprediksi oleh model CNN.")

model = load_model('fashion_mnist_cnn.h5')

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_container_width=True)
    st.write("Memprediksi...")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]  # shape (10,)

    # ---------- Ambil Top-3 ----------
    top3_idx = np.argsort(prediction)[::-1][:3]  # indeks terbesar ke terkecil
    top3_prob = prediction[top3_idx]
    top3_label = [class_names[i] for i in top3_idx]

    # ---------- Tampilkan Hasil ----------
    st.success("Top-3 Prediksi:")
    for rank, (label, prob) in enumerate(zip(top3_label, top3_prob), start=1):
        st.write(f"{rank}. **{label}** – Keyakinan: **{prob*100:.2f}%**")