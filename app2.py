import os, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU only
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from PIL import Image
import plotly.graph_objects as go

from interface.inference import infer
from interface import explanations
from interface import recipes

from google.oauth2 import service_account

GOOGLE_CLOUD_PROJECT = st.secrets['GOOGLE_CLOUD_PROJECT']
GOOGLE_CLOUD_LOCATION = st.secrets['GOOGLE_CLOUD_LOCATION']
GOOGLE_GENAI_USE_VERTEXAI = st.secrets['GOOGLE_GENAI_USE_VERTEXAI']

service_account = json.loads(st.secrets["GOOGLE_PRIVATE_KEY_JSON"])

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
tmp.write(json.dumps(service_account).encode())
tmp.close()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
os.environ["GOOGLE_CLOUD_PROJECT"] = service_account["project_id"]

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

/* Targets the <p> tag within the div that holds the st.slider label */
div[class*="stSlider"] p {
    font-size: 20px; /* Adjust the size as needed */
}

/* Target only the button with key="run_pred" */
.st-key-run_pred button {
    padding: 0.9rem 1.2rem !important;
    line-height: 0.8 !important;
}

/* In some Streamlit versions, the label is inside spans */
.st-key-run_pred button * {
    font-size: 22px !important;
    font-weight: 600 !important;
}

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

PATHOGEN_MODELS = {
    "E. coli":      {"Tmin": -1.0, "b": 0.018, "N0": 1e1, "Nmax": 1e9},
    "Listeria":     {"Tmin": -2.0, "b": 0.017, "N0": 1e1, "Nmax": 1e9},
    "Salmonella":   {"Tmin": -2.0, "b": 0.020, "N0": 1e1, "Nmax": 1e9},
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

def risk_from_count(N):
    # same logic as your classify_risk thresholds
    if N < 1e4:
        return "Safe", 0.2
    elif N < 1e6:
        return "Caution", 0.6
    else:
        return "High Risk", 0.95

def make_gauge(title, N):
    status, frac = risk_from_count(N)

    # map risk fraction to a 0-100 gauge
    value = int(frac * 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        title={"text": f"{title}<br><span style='font-size:12px'>Pred: {N:,.0f} CFU/g • {status}</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 35], "color": "#E8F8F5"},   # safe zone
                {"range": [35, 75], "color": "#FCF3CF"}, # caution zone
                {"range": [75, 100], "color": "#F5B7B1"} # high risk
            ],
            "threshold": {"line": {"color": "#E14F3D", "width": 4}, "thickness": 0.8, "value": value}
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def time_to_reach_threshold_logistic(N0, Nmax, mu, N_thresh):
    """
    Solve logistic growth for t when N(t) = N_thresh.
    Returns np.inf if mu==0 or if threshold not reachable.
    """
    if mu <= 0:
        return np.inf
    if N_thresh <= N0:
        return 0.0
    if N_thresh >= Nmax:
        return np.inf

    # Logistic: N(t) = Nmax / (1 + A*exp(-mu*t)), A = (Nmax-N0)/N0
    A = (Nmax - N0) / N0
    # Rearranged:
    # t = -(1/mu) * ln( ((Nmax/N_thresh)-1) / A )
    inside = ((Nmax / N_thresh) - 1.0) / A
    if inside <= 0:
        return np.inf
    return float(-(1.0 / mu) * np.log(inside))

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

col1, col2 = st.columns([1,3])

with col1:
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    input_method = st.radio("Food identification method:", ["Select Food Category","Upload Image"])

    if input_method == "Upload Image":
        if st.session_state.uploaded_file is None:

            uploaded = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

            if uploaded is not None:
                st.session_state.uploaded_file = uploaded
                st.rerun()   # immediately refresh UI

        else:
            img = Image.open(st.session_state.uploaded_file)
            st.image(img, width=150)

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
                    st.success(f"Detected dish: {food}")

            if st.button("🔄 Upload another image"):
                st.session_state.uploaded_file = None
                st.rerun()

    else:
        food = st.selectbox("Select food type:", list(FOOD_MODELS.keys()))

with col2:
    temperature = st.slider("🌡Storage Temperature (°C)", -5, 40, 4)
    time_label = st.select_slider("🕘Storage Time",
                            options=list(STORAGE_TIMES.keys()),
                            value="24 h" if "24 h" in STORAGE_TIMES else "1 d")
    time_hours = STORAGE_TIMES[time_label]

    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    if "prediction" not in st.session_state:
        st.session_state.prediction = {}

    if st.button("🍽 Should I eat it?",
                 type="primary",
                 use_container_width=True,
                 key="run_pred"):
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

st.markdown("---")

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

    # -----------------------------
    # DASHBOARD: Pathogen risk gauges + shelf-life
    # -----------------------------
    st.markdown("## 📊 Microbial Risk Dashboard")

    # Predict each pathogen using its own model parameters
    pathogen_counts = {}
    for name, pm in PATHOGEN_MODELS.items():
        mu_p = ratkowsky_growth_rate(temperature, pm["Tmin"], pm["b"])
        N_p = logistic_growth(pm["N0"], pm["Nmax"], mu_p, time_hours)
        pathogen_counts[name] = (N_p, mu_p)

    # Total count: simple conservative choice = max of pathogens (or sum if you prefer)
    total_count = max(v[0] for v in pathogen_counts.values())

    # --- Gauges in 4 columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.plotly_chart(make_gauge("E. coli", pathogen_counts["E. coli"][0]), use_container_width=True)
    with c2:
        st.plotly_chart(make_gauge("Listeria", pathogen_counts["Listeria"][0]), use_container_width=True)
    with c3:
        st.plotly_chart(make_gauge("Salmonella", pathogen_counts["Salmonella"][0]), use_container_width=True)
    with c4:
        st.plotly_chart(make_gauge("Total Count", total_count), use_container_width=True)

    # -----------------------------
    # Remaining shelf-life section
    # -----------------------------
    st.markdown("## ⏳ Remaining Shelf-Life")

    # Choose shelf-life limit: you can tune this
    # "Safe" limit = 1e4; "High risk" limit = 1e6
    SAFE_LIMIT = 1e4
    RISK_LIMIT = 1e6

    # Use the "total count driver" for shelf-life (conservative = worst mu among pathogens)
    worst_mu = max(mu for (_, mu) in pathogen_counts.values())

    # For shelf-life math, we need N0/Nmax; use a representative set
    N0_rep = 1e1
    Nmax_rep = 1e9

    t_safe = time_to_reach_threshold_logistic(N0_rep, Nmax_rep, worst_mu, SAFE_LIMIT)
    t_risk = time_to_reach_threshold_logistic(N0_rep, Nmax_rep, worst_mu, RISK_LIMIT)

    remaining_safe = max(0.0, t_safe - time_hours)
    remaining_risk = max(0.0, t_risk - time_hours)

    # Display
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Remaining time until 'Caution' (10⁴ CFU/g)", f"{remaining_safe:.1f} h")
    with m2:
        st.metric("Remaining time until 'High Risk' (10⁶ CFU/g)", f"{remaining_risk:.1f} h")
    with m3:
        st.metric("Current total estimate", f"{total_count:,.0f} CFU/g")

    # Optional progress bar toward high risk
    if np.isfinite(t_risk) and t_risk > 0:
        progress = min(1.0, time_hours / t_risk)
        st.progress(progress)
        st.caption(f"Progress toward high-risk limit at {temperature}°C: {progress*100:.0f}%")
    else:
        st.info("At this temperature, growth is minimal (or model predicts threshold not reachable).")

    # -----------------------------
    # Bacterial growth chart
    # -----------------------------
    # if p["fig"] is not None:
    #     st.plotly_chart(p["fig"])
    # else:
    #     st.warning("No figure returned by infer().")

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
