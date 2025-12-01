import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Automated Ophthalmologic Diagnosis", layout="centered")

# ---------------------------------------------------
# MEDICAL BACKGROUND + UI STYLING
# ---------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/6/65/Medical_Stethoscope.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 900;
        color: #003049;
        text-shadow: 2px 2px 6px white;
        margin-bottom: 15px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #003049;
        background: rgba(255,255,255,0.85);
        padding: 10px 16px;
        border-radius: 10px;
        display: inline-block;
    }

    .box {
        background: rgba(255,255,255,0.9);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #aac8d8;
        margin-top: 20px;
    }

    .result {
        background: #d1ecf1;
        padding: 12px;
        border-radius: 8px;
        border-left: 6px solid #0c5460;
        color: #0c5460;
        font-weight: 600;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE + SUBTITLE
# ---------------------------------------------------
st.markdown("<h1 class='title'>👁️ Automated Ophthalmologic Diagnosis</h1>", unsafe_allow_html=True)

st.markdown(
    "<p class='subtitle'>Upload a retina image to detect Cataract, Diabetic Retinopathy, Glaucoma, or Normal Eye.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# MODEL ARCHITECTURE (MATCH TRAINING CODE)
# ---------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3)
)
base_model.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(4, activation="softmax")

inputs = tf.keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ---------------------------------------------------
# LOAD WEIGHTS
# ---------------------------------------------------
try:
    model.load_weights("eye_disease_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# ---------------------------------------------------
# PREPROCESS IMAGE
# ---------------------------------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = ImageOps.fit(img, (256, 256))
    return np.expand_dims(np.array(img).astype("float32"), 0)

# ---------------------------------------------------
# UPLOAD IMAGE
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
if uploaded_file:
    st.markdown("<div class='box'>", unsafe_allow_html=True)

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Retina Image", use_container_width=True)

    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    probs = preds[0]

    pred_class = CLASS_NAMES[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    st.markdown(
        f"<div class='result'>Prediction: {pred_class.upper()} — {confidence:.2f}% Confidence</div>",
        unsafe_allow_html=True,
    )

    st.write("### Class Probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        st.write(f"- **{cls}**: {p*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<p style='text-align:center; font-size:18px; margin-top:25px;
color:#003049; font-weight:600;'>
Developed by <b>Ratnaprava Mohapatra</b>
</p>
""", unsafe_allow_html=True)
