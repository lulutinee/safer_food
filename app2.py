import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Scientific Food Microbial Risk Predictor",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
.main-title {font-size:38px; font-weight:700; color:#1F4E79;}
.safe {color:#1E8449; font-weight:bold;}
.warning {color:#D68910; font-weight:bold;}
.danger {color:#C0392B; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FOOD PARAMETERS
# Tmin and Ratkowsky constant b
# -----------------------------
FOOD_MODELS = {
    "Raw Poultry": {"Tmin": -2, "b": 0.02, "pathogen": "Salmonella"},
    "Raw Beef": {"Tmin": -1, "b": 0.018, "pathogen": "E. coli"},
    "Cooked Food": {"Tmin": 0, "b": 0.015, "pathogen": "Bacillus cereus"},
    "Dairy Product": {"Tmin": -2, "b": 0.017, "pathogen": "Listeria"},
    "Seafood": {"Tmin": -1, "b": 0.02, "pathogen": "Vibrio"},
    "Fresh Vegetables": {"Tmin": 0, "b": 0.014, "pathogen": "Listeria"}
}

# -----------------------------
# LOAD IMAGE CLASSIFICATION MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224,224,3)
    )
    return model

model_ai = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224,224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array


LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels = np.array(open(tf.keras.utils.get_file("labels.txt", LABELS_URL)).read().splitlines())

def map_food_category(predicted_label):

    label = predicted_label.lower()

    if any(x in label for x in ["chicken", "hen", "turkey"]):
        return "Raw Poultry"

    elif any(x in label for x in ["beef", "steak", "burger"]):
        return "Raw Beef"

    elif any(x in label for x in ["fish", "salmon", "tuna"]):
        return "Seafood"

    elif any(x in label for x in ["milk", "cheese", "yogurt"]):
        return "Dairy Product"

    elif any(x in label for x in ["broccoli","salad","cucumber","vegetable"]):
        return "Fresh Vegetables"

    else:
        return "Cooked Food"

# -----------------------------
# SCIENTIFIC FUNCTIONS
# -----------------------------

def ratkowsky_growth_rate(T, Tmin, b):
    if T <= Tmin:
        return 0
    return (b * (T - Tmin))**2

def logistic_growth(N0, Nmax, mu, t):
    return Nmax / (1 + ((Nmax - N0)/N0)*np.exp(-mu*t))

def classify_risk(N):
    if N < 1e4:
        return "Safe", "safe"
    elif 1e4 <= N < 1e6:
        return "Caution", "warning"
    else:
        return "High Risk", "danger"

# -----------------------------
# UI
# -----------------------------

st.markdown('<div class="main-title">🧪 Scientific Microbial Growth Predictor</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    input_method = st.radio("Food identification method:",
                            ["Select Food Category", "Upload Image"])

if input_method == "Upload Image":

    file = st.file_uploader("Upload food image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_column_width=True)


        with st.spinner("Analyzing food with AI..."):

            processed = preprocess_image(img)

            with tf.device("/CPU:0"):
                preds = model_ai(processed, training=False).numpy()

            decoded = decode_predictions(preds, top=1)[0][0]

            predicted_label = decoded[1]
            confidence = decoded[2]

        st.success(f"Detected food: {predicted_label}")
        st.write(f"Confidence: {confidence:.2%}")

        detected_food = map_food_category(predicted_label)
        food = detected_food
else:
    food = st.selectbox("Select food type:", list(FOOD_MODELS.keys()))


with col2:
    temperature = st.slider("Storage Temperature (°C)", -5, 40, 4)
    time_hours = st.slider("Storage Time (hours)", 0, 240, 24)

st.markdown("---")

if st.button("🔬 Run Scientific Prediction"):

    model = FOOD_MODELS[food]

    Tmin = model["Tmin"]
    b = model["b"]
    pathogen = model["pathogen"]

    # Assumptions
    N0 = 1e2      # Initial contamination 100 CFU/g
    Nmax = 1e9    # Maximum bacterial load

    mu = ratkowsky_growth_rate(temperature, Tmin, b)
    final_count = logistic_growth(N0, Nmax, mu, time_hours)

    status, level = classify_risk(final_count)

    st.markdown(f"## Status: <span class='{level}'>{status}</span>", unsafe_allow_html=True)

    st.write(f"### Estimated Bacterial Load: {final_count:,.0f} CFU/g")
    st.write(f"### Predominant Pathogen of Concern: {pathogen}")

    if status == "Safe":
        st.success("Microbial growth remains below hazardous levels.")
    elif status == "Caution":
        st.warning("Microbial levels are approaching infectious dose. Cook thoroughly (>75°C).")
    else:
        st.error("Bacterial load exceeds safe limits. High foodborne illness risk.")

    # -----------------------------
    # Growth Curve
    # -----------------------------
    times = np.linspace(0, time_hours, 100)
    growth_curve = logistic_growth(N0, Nmax, mu, times)

    fig, ax = plt.subplots()
    ax.plot(times, growth_curve)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CFU/g")
    ax.set_yscale("log")
    ax.set_title("Predicted Microbial Growth Curve")
    st.pyplot(fig)

    # -----------------------------
    # Scientific Explanation
    # -----------------------------
    st.markdown("### 📘 Scientific Explanation")
    st.write("""
This prediction uses:

• Ratkowsky square-root temperature model to estimate bacterial growth rate
• Logistic growth model to simulate microbial population dynamics
• Assumed initial contamination of 10² CFU/g
• Maximum carrying capacity of 10⁹ CFU/g

Risk classification is based on commonly accepted infectious dose ranges.
""")

    # -----------------------------
    # Recipe Guidance
    # -----------------------------
    st.markdown("### 👨‍🍳 Recommendations")

    if status == "Safe":
        st.info("Suitable for standard cooking methods (grilling, baking, sautéing).")
    elif status == "Caution":
        st.info("Cook thoroughly to internal temperature ≥ 75°C. Avoid raw consumption.")
    else:
        st.info("Discard the product. Do not attempt to cook or salvage.")
