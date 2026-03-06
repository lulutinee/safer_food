import os, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU only
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from PIL import Image
import plotly.graph_objects as go

#GOOGLE_CLOUD_PROJECT = st.secrets['GOOGLE_CLOUD_PROJECT']
#GOOGLE_CLOUD_LOCATION = st.secrets['GOOGLE_CLOUD_LOCATION']
#GOOGLE_GENAI_USE_VERTEXAI = st.secrets['GOOGLE_GENAI_USE_VERTEXAI']
from google.oauth2 import service_account
# service_account = json.loads(st.secrets["GOOGLE_PRIVATE_KEY_JSON"])

# tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
# tmp.write(json.dumps(service_account).encode())
# tmp.close()

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
# os.environ["GOOGLE_CLOUD_PROJECT"] = service_account["project_id"]

from interface.inference import infer
from interface import explanations, recipes
import thermometer_component
from thermometer_component import thermometer_slider

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
import streamlit as st

st.markdown(
"""
<style>

/* =========================
   App Background
   ========================= */

.stApp {
    background-color: #2C2A29;
    color: #F5F5F5;
}

/* =========================
   Text
   ========================= */

.sf-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: -0.7rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #F5F5F5;
}

p, label {
    color: #F5F5F5;
}

/* =========================
   Buttons
   ========================= */

div.stButton > button {
    background-color: #E14F3D;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    padding: 0.6em 1.2em;
}

div.stButton > button:hover {
    background-color: #c94333;
    color: white;
}

/* =========================
   Selectbox
   ========================= */

div[data-baseweb="select"] > div {
    background-color: #161514 !important;
    border: 1px solid #FFFFFF !important;
    color: white !important;
}

/* Dropdown menu */

ul[role="listbox"] {
    background-color: #161514 !important;
}

ul[role="listbox"] li {
    color: white !important;
}

/* =========================
   Number input
   ========================= */

div[data-baseweb="input"] > div {
    background-color: #161514 !important;
    border: 1px solid white !important;
}

/* =========================
   Slider
   ========================= */

.stSlider > div[data-baseweb="slider"] > div > div {
    background: #E14F3D;
}

/* =========================
   Tabs
   ========================= */

button[data-baseweb="tab"] {
    background-color: #161514;
    color: #F5F5F5;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #E14F3D;
}

/* =========================
   Metric cards
   ========================= */

div[data-testid="stMetric"] {
    background-color: #1d1c1b;
    padding: 10px;
    border-radius: 10px;
}

/* =========================
   Dataframe
   ========================= */

[data-testid="stDataFrame"] {
    background-color: #161514;
}

/* =========================
   Sidebar
   ========================= */

section[data-testid="stSidebar"] {
    background-color: #121110;
}

/* =========================
   File uploader
   ========================= */

section[data-testid="stFileUploader"] {
    background-color: #1d1c1b;
    border-radius: 10px;
    padding: 10px;
}

/* =========================
   Expander
   ========================= */

details {
    background-color: #1d1c1b;
    border-radius: 8px;
    padding: 8px;
}

/* =========================
   Scrollbar (optional)
   ========================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #E14F3D;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #161514;
}

/* ---------------------------
   HARD OVERRIDE for Streamlit 1.54.0 NumberInput steppers
   Scoped ONLY to these two inputs via .st-key-...
   --------------------------- */


/* Keep the text readable */
.st-key-days_input  input,
.st-key-hours_input input {
  color: #F5F5F5 !important;
  font-weight: 800 !important;
}


/* Arrow buttons background + remove default styles */
.st-key-days_input  button,
.st-key-hours_input button {
  background-color: #161514 !important;
  border: none !important;
  box-shadow: none !important;
}

/* Arrow icons color */
.st-key-days_input  button svg,
.st-key-hours_input button svg {
  fill: #E14F3D !important;
  color: #E14F3D !important;
}

/* Hover effect */
.st-key-days_input  button:hover,
.st-key-hours_input button:hover {
  background-color: rgba(225,79,61,0.12) !important;
}


</style>
""",
unsafe_allow_html=True
)

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
# Thermometer slider
# -----------------------------
def thermometer_slider(
    label: str,
    min_value: int = -5,
    max_value: int = 40,
    value: int = 4,
    step: int = 1,
    height: int = 360,
    key: str = "thermo_temp",
    color: str = "#E14F3D",
):
    """
    Vertical thermometer slider rendered in HTML.
    User drags mercury level inside thermometer to change value.
    Returns an int (or float if you change it) back to Streamlit.
    """
    # Streamlit component returns a value; we keep state in session_state
    if key not in st.session_state:
        st.session_state[key] = value

    # Use the last value as default (so it persists on reruns)
    value = st.session_state[key]

    html = f"""
    <style>
      .thermo-wrap {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        display: flex;
        flex-direction: column;
        gap: 10px;
        user-select: none;
      }}
      .thermo-label {{
        font-size: 20px;
        font-weight: 600;
        color: #fff;
      }}
      .thermo-row {{
        display:flex;
        align-items:center;
        gap: 16px;
      }}
      .thermo {{
        position: relative;
        width: 70px;
        height: 300px;
      }}
      .tube {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: 26px;
        height: 240px;
        top: 10px;
        border-radius: 18px;
        border: 3px solid #444;
        background: linear-gradient(180deg, #f7f7f7 0%, #ededed 100%);
        overflow: hidden;
      }}
      .bulb {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 0px;
        width: 54px;
        height: 54px;
        border-radius: 50%;
        border: 3px solid #444;
        background: linear-gradient(180deg, #f7f7f7 0%, #eaeaea 100%);
        overflow: hidden;
      }}
      .mercury {{
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 0%;
        background: {color};
      }}
      .mercury-gloss {{
        position: absolute;
        left: 18%;
        width: 18%;
        top: 4%;
        bottom: 4%;
        border-radius: 99px;
        background: rgba(255,255,255,0.35);
      }}
      .bulb .mercury {{
        border-radius: 50%;
      }}
      .value-box {{
        min-width: 90px;
        font-size: 20px;
        font-weight: 800;
        color: #fff;
      }}
      .hint {{
        font-size: 12px;
        color: #666;
        margin-top: 2px;
      }}

      /* Invisible input overlay to capture drag inside thermometer */
      .overlay {{
        position:absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 10px;
        width: 26px;
        height: 240px;
        border-radius: 18px;
        cursor: ns-resize;
        background: rgba(0,0,0,0);
      }}

      /* Tick marks (optional) */
      .ticks {{
        position:absolute;
        left: calc(50% + 18px);
        top: 10px;
        height: 240px;
        width: 16px;
        display:flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .tick {{
        height: 2px;
        background: #555;
        border-radius: 2px;
        width: 10px;
        opacity: 0.7;
      }}
    </style>

    <div class="thermo-wrap">
      <div class="thermo-label">{label}</div>

      <div class="thermo-row">
        <div class="thermo" id="thermo">
          <div class="tube">
            <div class="mercury" id="tubeMercury"></div>
            <div class="mercury-gloss"></div>
          </div>

          <div class="bulb">
            <div class="mercury" id="bulbMercury"></div>
            <div class="mercury-gloss"></div>
          </div>

          <div class="overlay" id="dragArea" aria-label="drag temperature"></div>

          <div class="ticks">
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
            <div class="tick"></div>
          </div>
        </div>

        <div>
          <div class="value-box"><span id="valText">{value}</span> °C</div>
          <div class="hint">Drag inside the thermometer</div>
        </div>
      </div>
    </div>

    <script>
      const minV = {min_value};
      const maxV = {max_value};
      const step = {step};
      let value = {value};

      const tubeMercury = document.getElementById("tubeMercury");
      const bulbMercury = document.getElementById("bulbMercury");
      const valText = document.getElementById("valText");
      const dragArea = document.getElementById("dragArea");

      function clamp(x, a, b) {{
        return Math.max(a, Math.min(b, x));
      }}

      function snapToStep(v) {{
        const snapped = Math.round((v - minV) / step) * step + minV;
        return clamp(snapped, minV, maxV);
      }}

      function percentFromValue(v) {{
        return ((v - minV) / (maxV - minV)) * 100;
      }}

      function setUI(v) {{
        const pct = percentFromValue(v);
        tubeMercury.style.height = pct + "%";
        bulbMercury.style.height = "100%";
        valText.textContent = v;
      }}

      function emitValue(v) {{
        // Streamlit component protocol
        const out = {{ value: v }};
        window.parent.postMessage({{
          isStreamlitMessage: true,
          type: "streamlit:setComponentValue",
          value: out
        }}, "*");
      }}

      function updateFromClientY(clientY) {{
        const rect = dragArea.getBoundingClientRect();
        const y = clamp(clientY, rect.top, rect.bottom);
        const rel = (rect.bottom - y) / rect.height; // 0 bottom -> 1 top
        let v = minV + rel * (maxV - minV);
        v = snapToStep(v);
        value = v;
        setUI(value);
        emitValue(value);
      }}

      let dragging = false;

      dragArea.addEventListener("mousedown", (e) => {{
        dragging = true;
        updateFromClientY(e.clientY);
      }});

      window.addEventListener("mousemove", (e) => {{
        if (!dragging) return;
        updateFromClientY(e.clientY);
      }});

      window.addEventListener("mouseup", () => {{
        dragging = false;
      }});

      // Touch support
      dragArea.addEventListener("touchstart", (e) => {{
        dragging = true;
        updateFromClientY(e.touches[0].clientY);
      }}, {{passive:true}});

      window.addEventListener("touchmove", (e) => {{
        if (!dragging) return;
        updateFromClientY(e.touches[0].clientY);
      }}, {{passive:true}});

      window.addEventListener("touchend", () => {{
        dragging = false;
      }});

      // init UI
      setUI(value);
      emitValue(value);
    </script>
    """

    # Return value from component
    result = components.html(html, height=height)

    # result comes as dict {"value": x} or None
    if isinstance(result, dict) and "value" in result:
        st.session_state[key] = int(result["value"])

    return st.session_state[key]

# -----------------------------
# Digital timer
# -----------------------------
def days_hours_input(
    label="Storage Time",
    default_days=1,
    default_hours=0,
    max_days=21,
    key="storage_time"
):

    if f"{key}_days" not in st.session_state:
        st.session_state[f"{key}_days"] = default_days

    if f"{key}_hours" not in st.session_state:
        st.session_state[f"{key}_hours"] = default_hours

    st.markdown(f"### ⏱️ {label}")

    col_days, col_hours = st.columns(2)

    with col_days:
        days = st.number_input(
            "Days",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
            key="days_input"
        )

    with col_hours:
        hours = st.number_input(
            "Hours",
            min_value=0,
            max_value=23,
            value=0,
            step=1,
            key="hours_input"
        )

    # -------- Return total hours --------
    time_hours = int(days) * 24 + int(hours)

    st.caption(f"Total time: **{int(days)}d {int(hours)}h**  →  **{time_hours} hours**")

    return time_hours

# -----------------------------
# Streamlit UI
# -----------------------------

# Header layout: logo left, title right
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    st.image(logo)

with col_title:
    st.markdown('<div class="sf-title"> Is it safe to eat?</div>', unsafe_allow_html=True, text_alignment='center')
    st.caption("SaferFood™ — Microbial risk estimates based on predictive microbiology models.", text_alignment='center')

with st.sidebar:
    st.image(logo)
    st.markdown("### SaferFood™")
    st.caption("AI-assisted food safety estimation")

col1, col2, col3, col4  = st.columns([1,1,1,1], vertical_alignment="center")

with col1:
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    input_method = st.radio("Food identification method:", ["Select Food Category","Upload Image"])

    if input_method == "Upload Image":
        if st.session_state.uploaded_file is None:

            st.markdown("""
                <style>
                /* Change the background color of the drag-and-drop area */
                [data-testid='stFileUploaderDropzone'] {
                    background-color: #161514; /* Red background */
                    color: #ffffff;             /* White text color */
                }
                </style>
                """, unsafe_allow_html=True)
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
        st.markdown("""
        <style>
        .stSelectbox div[data-baseweb="select"] > div:first-child {
            background-color: #161514; /* Use your desired color */
        }
        </style>
        """, unsafe_allow_html=True)
        food = st.selectbox("Select food type:", list(FOOD_MODELS.keys()), width=200)

with col2:
    #temperature = st.slider("🌡Storage Temperature (°C)", -5, 40, 4)
    temperature = thermometer_slider(
        "Storage Temperature (°C)",
        min_value=-5,
        max_value=40,
        value=4,
        step=1,
        color="#E14F3D",
        key="temp"
    )

    st.write("Selected temperature:", temperature, "°C")

with col3:
    time_hours = days_hours_input(
        label="Storage Time",
        default_days=1,
        default_hours=0
    )

    st.write("Selected storage time:", time_hours, "hours")

    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    if "prediction" not in st.session_state:
        st.session_state.prediction = {}

with col4:
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button("🍽 Should I eat it?",
                    type="primary",
                    width=220,
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

    if p["cooking_reco"] == "raw" or "medium":
        status = "✅ Safe"

        if p["cooking_reco"] == "high":
            status = "⚠️ Caution"
    else:
        status = "☠️❌ High risk"
    st.markdown(f"## STATUS: {status}")
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
    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")

    with c1:
        st.markdown(f'Please consider reaching an internal temperature of 71°C for ground meats or 74°C for poultry!')
        st.plotly_chart(make_gauge("E. coli", pathogen_counts["E. coli"][0]))

    with c2:
        st.plotly_chart(make_gauge("Listeria", pathogen_counts["Listeria"][0]))
    with c3:
        st.plotly_chart(make_gauge("Salmonella", pathogen_counts["Salmonella"][0]))
    with c4:
        st.plotly_chart(make_gauge("Total Count", total_count))

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
