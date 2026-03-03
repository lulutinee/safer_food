import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU only
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from PIL import Image

from interface.inference import infer
from interface import explanations
from interface import recipes

# -----------------------------
# Page Config
# -----------------------------
LOGO_PATH = "SaferFood_logo.png"  # <- your file name

logo = Image.open(LOGO_PATH)

LOGO2_PATH = "logo4.PNG"  # <- your file name

logo2 = Image.open(LOGO2_PATH)

st.set_page_config(page_title="SaferFood", page_icon=logo2, layout="wide")
# st.set_page_config(
#     page_title="AI Food Safety Predictor",
#     page_icon="🧪",
#     layout="wide"
# )

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {font-size:38px; font-weight:700; color:white;}
.safe {color:#1E8449; font-weight:bold;}
.warning {color:#D68910; font-weight:bold;}
.danger {color:#C0392B; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Variables
# -----------------------------
#Storage times
STORAGE_TIMES = {}
# Hours
for h in range(0, 24):
    STORAGE_TIMES[f"{h} h"] = h
# Days
for d in range(1, 22):
    STORAGE_TIMES[f"{d} d"] = d * 24

# -----------------------------
# Food Models
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
# Food-101 Labels
# -----------------------------
FOOD101_LABELS = [
"apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
"beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
"bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
"ceviche","cheesecake","cheese_plate","chicken_curry","chicken_quesadilla",
"chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
"club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
"deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots",
"falafel","filet_mignon","fish_and_chips","foie_gras","french_fries",
"french_onion_soup","french_toast","fried_calamari","fried_rice",
"frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich",
"grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup",
"hot_dog","huevos_rancheros","hummus","ice_cream","lasagna","lobster_bisque",
"lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup",
"mussels","nachos","omelette","onion_rings","oysters","pad_thai",
"paella","pancakes","panna_cotta","peking_duck","pho","pizza",
"pork_chop","poutine","prime_rib","pulled_pork_sandwich","ramen",
"ravioli","red_velvet_cake","risotto","samosa","sashimi","scallops",
"seaweed_salad","shrimp_and_grits","spaghetti_bolognese","spaghetti_carbonara",
"spring_rolls","steak","strawberry_shortcake","sushi","tacos",
"takoyaki","tiramisu","tuna_tartare","waffles"
]

# -----------------------------
# Load Food AI Model
# -----------------------------
@st.cache_resource
def load_food_model():
    model = tf.keras.models.load_model("tf_model.h5")
    return model

food_model = load_food_model()

# -----------------------------
# Scientific Functions
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

# Map AI label to microbial category
def map_food_category(label):
    label = label.lower()
    poultry = ["chicken","turkey","wings"]
    beef = ["beef","burger","steak","meatball"]
    seafood = ["fish","salmon","tuna","shrimp","sushi"]
    dairy = ["cheese","ice_cream","yogurt","cheesecake"]
    vegetables = ["salad","vegetable","broccoli","ratatouille"]

    if any(x in label for x in poultry):
        return "Raw Poultry"
    elif any(x in label for x in beef):
        return "Raw Beef"
    elif any(x in label for x in seafood):
        return "Seafood"
    elif any(x in label for x in dairy):
        return "Dairy Product"
    elif any(x in label for x in vegetables):
        return "Fresh Vegetables"
    else:
        return "Cooked Food"

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_food_image(img_pil):
    img = img_pil.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Streamlit UI
# -----------------------------

# Header layout: logo left, title right
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    st.image(logo, use_container_width=True)

with col_title:
    st.markdown('<div class="main-title"> Is it safe to eat?</div>', unsafe_allow_html=True, text_alignment='center')
    st.caption("SaferFood™ — Microbial risk estimates based on predictive microbiology models.", text_alignment='center')

with st.sidebar:
    st.image(logo, use_container_width=True)
    st.markdown("### SaferFood™")
    st.caption("AI-assisted food safety estimation")

col1, col2 = st.columns([1,2])

with col1:
    input_method = st.radio("Food identification method:", ["Select Food Category","Upload Image"])

    if input_method == "Upload Image":
        file = st.file_uploader("Upload food image", type=["jpg","png","jpeg"])
        if file:
            img = Image.open(file)
            st.image(img)

            with st.spinner("Analyzing food with AI..."):
                processed = preprocess_food_image(img)
                preds = food_model(processed, training=False).numpy()

                # Safety check
                if np.isnan(preds).any():
                    st.error("Prediction produced NaN. Check preprocessing or model.")
                else:
                    top_index = np.argmax(preds[0])
                    food = FOOD101_LABELS[top_index]
                    confidence = preds[0][top_index]
                    st.success(f"Detected dish: {food} ({confidence:.2%})")

    else:
        food = st.selectbox("Select food type:", list(FOOD_MODELS.keys()))

with col2:
    temperature = st.slider("🌡Storage Temperature (°C)", -5, 40, 4)
    time_label = st.select_slider("🕘Storage Time (hours)",
                            options=list(STORAGE_TIMES.keys()),
                            value="24 h" if "24 h" in STORAGE_TIMES else "1 d")
    time_hours = STORAGE_TIMES[time_label]

st.markdown("---")

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = {}

if st.button("🍽 Should I eat it?"):
    params = {
        "matrixID": food,
        "temperature": temperature,
        "time": time_hours
    }

    results = infer(params=params)
    is_safe = results.get('is_safe')
    bacterias = results.get('bacterias')
    cooking_reco = results.get('cooking_reco')
    fig = results.get('fig')

    # Store everything needed for display + AI explanation
    st.session_state.prediction_done = True
    st.session_state.prediction = {
        "food": food,
        "temperature": temperature,
        "time_hours": time_hours,
        "bacterias": bacterias,
        "is_safe": is_safe,
        "cooking_reco": cooking_reco,
        "fig": fig
    }

if st.session_state.prediction_done:
    p = st.session_state.prediction

    if p["is_safe"] is True:
        st.markdown("# YOUR FOOD IS SAFE TO EAT !")
    else:
        st.markdown("# Were you really considering eating this?")
        st.markdown('Seriously. There is a risk of:')
        st.markdown("\n".join(f"- {bacteria}" for bacteria in p['bacterias']))

    if p["cooking_reco"] is not None:
        st.markdown(f'Given our predictions, your {p["food"]} should be eaten at least {p["cooking_reco"]}.')
        st.markdown(f'Please consider reaching an internal temperature of 71°C for ground meats or 74°C for poultry!')

    if p["fig"] is not None:
        st.plotly_chart(p["fig"])
    else:
        st.warning("No figure returned by infer().")

    if st.button("What does it mean?"):
        with st.spinner("Generating detailed explanation..."):
            explanations = explanations.risk_explanation(p["bacterias"], max_output_tokens=2000)

        st.markdown("### 🧠 AI Detailed Explanation")
        st.write(explanations)

    if st.button("Recipe suggestions"):
            with st.spinner("Looking for yummy recipes"):
                recipes = recipes.recipe_suggestion(ingredient=p["food"], cooking=p["cooking_reco"], provider='auto', max_output_tokens=5000)

            st.markdown(f'Given our predictions, the recommended cooking temperature for your {p["food"]} is: {p["cooking_reco"]}')
            st.markdown(recipes)
