import streamlit as st
from PIL import Image
import random
from io import BytesIO
import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT_PATH = Path(__file__).parent


MODEL = tf.keras.models.load_model(f"{ROOT_PATH}/saved_models/1")
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']


def predict(image):
    # Resize image to 256x256 (model input size)
    img_resized = image.resize((256, 256))

    # Convert to array
    img_array = np.array(img_resized) / 255.0  # optional normalization if model trained with it
    img_batch = np.expand_dims(img_array, 0)

    # Make prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions)
    return {'class': predicted_class, 'confidence': float(confidence)}


st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="centered"
)


st.title("🥔 Potato Disease Detection")
st.write("Upload a potato leaf image to detect disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width=300)  # adjust width as needed

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing..."):
            # Pass the PIL Image, not uploaded_file
            result = predict(image)
            label, confidence = result.get("class", None), result.get("confidence", 0.0)

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")

        # Recommendation messages
        if label == "Healthy":
            st.success("✅ The potato plant looks healthy. Keep monitoring regularly.")
        elif label == "Early_blight":
            st.warning("⚠️ Early Blight detected. Consider preventive measures.")
        elif label == "Late_blight":
            st.error("❌ Late Blight detected. Immediate action needed.")

st.markdown("---")
