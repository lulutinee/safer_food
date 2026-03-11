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

# GOOGLE_CLOUD_PROJECT = st.secrets['GOOGLE_CLOUD_PROJECT']
# GOOGLE_CLOUD_LOCATION = st.secrets['GOOGLE_CLOUD_LOCATION']
# GOOGLE_GENAI_USE_VERTEXAI = st.secrets['GOOGLE_GENAI_USE_VERTEXAI']
from google.oauth2 import service_account
# service_account = json.loads(st.secrets["GOOGLE_PRIVATE_KEY_JSON"])

# tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
# tmp.write(json.dumps(service_account).encode())
# tmp.close()

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
# os.environ["GOOGLE_CLOUD_PROJECT"] = service_account["project_id"]

from interface.inference import infer
from interface import explanations, recipes, bacteria_information
import thermometer_component
from thermometer_component import thermometer_slider
import re

# -----------------------------
# Page Config
# -----------------------------
LOGO_PATH = "SaferFood_logo.png"  # <- your file name

logo = Image.open(LOGO_PATH)

LOGO2_PATH = "logo4.PNG"  # <- your file name

logo2 = Image.open(LOGO2_PATH)

st.set_page_config(page_title="SaferFood", page_icon=logo2, layout="wide")

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
    "Poultry": {"Tmin": -2, "b": 0.02, "pathogen": "Salmonella", "food": "poultry"},
    "Beef": {"Tmin": -1, "b": 0.018, "pathogen": "E. coli", "food": "beef"},
    #"Cooked Food": {"Tmin": 0, "b": 0.015, "pathogen": "Bacillus cereus"},
    #"Dairy Product": {"Tmin": -2, "b": 0.017, "pathogen": "Listeria"},
    "Seafood": {"Tmin": -1, "b": 0.02, "pathogen": "Vibrio", "food": "seafood"},
    #"Fresh Vegetables": {"Tmin": 0, "b": 0.014, "pathogen": "Listeria"},
    "Pork": {"Tmin": -1, "b": 0.018, "pathogen": "E. coli", "food": "pork"}
}

PATHOGEN_MODELS = {
    "E. coli":      {"Tmin": -1.0, "b": 0.018, "N0": 1e1, "Nmax": 1e9},
    "Listeria":     {"Tmin": -2.0, "b": 0.017, "N0": 1e1, "Nmax": 1e9},
    "Salmonella":   {"Tmin": -2.0, "b": 0.020, "N0": 1e1, "Nmax": 1e9},
}

MICROORGANISM = bacteria_information.MICROORGANISM
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
"frozen_yogurt","garlic_bread","gnocchi","greek_salad","Quick cooking_cheese_sandwich",
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
def classify_risk(N, organism):
    thresholds = MICROORGANISM[organism]

    raw = thresholds["raw"]
    medium = thresholds["medium"]
    high = thresholds["high"]

    if N < medium:
        return "✅Safe", "safe"
    elif medium <= N < high:
        return "Caution", "warning"
    else:
        return "High Risk", "danger"

def risk_from_count(N, organism):
    # same logic as your classify_risk thresholds
    thresholds = MICROORGANISM[organism]

    raw = thresholds["raw"]
    medium = thresholds["medium"]
    high = thresholds["high"]

    if N <= raw:
        return "✅Safe", N/high
    if N < medium:
        return "Safe", N/high
    elif N < high:
        return "⚠️Caution", N/high
    else:
        return "❌High Risk", 1

def make_gauge(title, N, organism):
    """
    Gauge where N is already in log CFU/g.
    Axis goes from 0 to thresholds['high'].
    """

    thresholds = MICROORGANISM[organism]

    raw = thresholds["raw"]
    medium = thresholds["medium"]
    high = thresholds["high"]

    # N is already log CFU/g
    value = max(0, min(N, high))  # clamp to gauge range

    status, _ = risk_from_count(N, organism)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " log CFU"},
        title={
            "text": f"{title}<br><span style='font-size:22px'>{status}</span>"
        },
        gauge={
            "axis": {"range": [0,high]},
            "steps": [
                {"range": [0, raw], "color": "#E8F8F5"},       # safe
                {"range": [raw, medium], "color": "#FCF3CF"}, # caution
                {"range": [medium, high], "color": "#F5B7B1"} # high risk
            ],
            "threshold": {
                "line": {"color": "#E14F3D", "width": 4},
                "thickness": 0.8,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=10)
    )

    return fig

def make_log_cfu_gauge(title, N, organism):
    """
    Gauge uses log scale internally, but displays CFU/g labels.

    Parameters
    ----------
    N : float
        Predicted concentration in log CFU/g
    organism : str
        MICROORGANISM key
    """

    thresholds = MICROORGANISM[organism]

    raw = thresholds["raw"]
    medium = thresholds["medium"]
    high = thresholds["high"]

    status, _ = risk_from_count(N, organism)

    # clamp only for display
    gauge_value = max(0, min(N, high))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={
            "suffix": " log CFU/g",
            "font": {"size": 24}
        },
        title={
            "text": (
                f"{title}<br>"
                f"<span style='font-size:12px'>"
                f"{thresholds['usual_name']} • {status}<br>"
                f"Pred: {10**N:.2e} CFU/g"
                f"</span>"
            )
        },
        gauge={
            "axis": {
                "range": [0, high],
                "tickmode": "array",
                "tickvals": list(range(0, int(high) + 1)),
                "ticktext": [f"1e{i}" for i in range(0, int(high) + 1)]
            },
            "steps": [
                {"range": [0, raw], "color": "#E8F8F5"},
                {"range": [raw, medium], "color": "#FCF3CF"},
                {"range": [medium, high], "color": "#F5B7B1"},
            ],
            "threshold": {
                "line": {"color": "#E14F3D", "width": 4},
                "thickness": 0.8,
                "value": gauge_value
            },
            "bar": {"color": "#2C2A29"}
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig
# Map AI label to microbial category
def map_food_category(label):
    label = label.lower()
    poultry = ["chicken","turkey","wings","poultry","peking_duck"]
    beef = ["beef","burger","steak","meatball","prime_rib"]
    seafood = ["fish","salmon","tuna","shrimp","sushi", "ceviche", "grilled_salmon", "seafood"]
    #dairy = ["cheese","ice_cream","yogurt","cheesecake"]
    #vegetables = ["salad","vegetable","broccoli","ratatouille"]

    if any(x in label for x in poultry):
        return "poultry"
    elif any(x in label for x in beef):
        return "beef"
    elif any(x in label for x in seafood):
        return "seafood"
    #elif any(x in label for x in dairy):
        return "Dairy Product"
    #elif any(x in label for x in vegetables):
        return "Fresh Vegetables"
    else:
        return "pork"

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
    days_key = f"{key}_days"
    hours_key = f"{key}_hours"

    if days_key not in st.session_state:
        st.session_state[days_key] = default_days

    if hours_key not in st.session_state:
        st.session_state[hours_key] = default_hours

    st.markdown(f"### ⏱️ {label}")

    col_days, col_hours = st.columns(2)

    with col_days:
        days = st.number_input(
            "Days",
            min_value=0,
            max_value=max_days,
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
# Shelf-life gauge
# -----------------------------
def make_shelflife_gauge(remaining_hours, max_display_hours=72):
    """
    Shelf-life gauge for Streamlit dashboard.

    Parameters
    ----------
    remaining_hours : float | int | None
        Remaining safe time before high risk.
    max_display_hours : float
        Upper bound shown on the gauge.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    if remaining_hours is None:
        value = 0
        title = "Shelf-life unavailable"
        display_text = "No data"
    else:
        value = max(0, min(float(remaining_hours), max_display_hours))
        title = "Remaining Shelf-Life"
        display_text = f"{float(remaining_hours):.1f} h"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " h", "font": {"size": 28}},
        title={"text": f"{title}<br><span style='font-size:13px'>{display_text}</span>"},
        gauge={
            "axis": {"range": [0, max_display_hours]},
            "bar": {"color": "#E14F3D"},
            "steps": [
                {"range": [0, max_display_hours * 0.25], "color": "#F5B7B1"},   # red-ish
                {"range": [max_display_hours * 0.25, max_display_hours * 0.6], "color": "#FCF3CF"},  # orange/yellow
                {"range": [max_display_hours * 0.6, max_display_hours], "color": "#D5F5E3"},  # green
            ],
            "threshold": {
                "line": {"color": "#2C2A29", "width": 5},
                "thickness": 0.8,
                "value": value,
            },
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig

# -----------------------------
# Format recipe
# -----------------------------
def format_recipe_text(recipe_text):
    """
    Accepts recipe_text as either:
    - a string
    - a list of strings

    Returns a nicely formatted markdown recipe.
    """

    # 1) Normalize to a list of clean lines
    if isinstance(recipe_text, list):
        lines = [str(x).strip() for x in recipe_text if str(x).strip()]
    elif isinstance(recipe_text, str):
        lines = [x.strip() for x in recipe_text.split("\n") if x.strip()]
    else:
        return "## 🍽 Recipe\n\nRecipe format not recognized."

    title = ""
    description = ""
    body = []

    i = 0
    while i < len(lines):
        raw_line = lines[i].strip()
        line = raw_line.lower()

        # Remove markdown ### for matching
        normalized = re.sub(r"^#+\s*", "", line).strip()

        # ---------- TITLE ----------
        if normalized.startswith("recipe name"):
            # Case 1: "Recipe Name: Chicken"
            if ":" in raw_line:
                title = raw_line.split(":", 1)[1].strip()
            # Case 2: next line contains title
            elif i + 1 < len(lines):
                title = lines[i + 1].strip()
                i += 1
            i += 1
            continue

        # ---------- DESCRIPTION ----------
        if normalized.startswith("short description"):
            # Case 1: "Short Description: ...."
            if ":" in raw_line:
                description = raw_line.split(":", 1)[1].strip()
            # Case 2: next line contains description
            elif i + 1 < len(lines):
                description = lines[i + 1].strip()
                i += 1
            i += 1
            continue

        # ---------- BODY ----------
        body.append(raw_line)
        i += 1

    # Fallbacks if title missing
    if not title:
        # Try using first non-header line as title
        for line in lines:
            test = re.sub(r"^#+\s*", "", line).strip().lower()
            if not any(
                test.startswith(x)
                for x in ["recipe name", "short description", "key ingredients", "cooking method", "basic preparation steps"]
            ):
                title = line.strip("*# ").strip()
                break

    if not title:
        title = "Recipe"

    # Remove duplicated title/description from body if they slipped through
    cleaned_body = []
    for line in body:
        check = re.sub(r"^#+\s*", "", line).strip()
        if check == title or check == description:
            continue
        cleaned_body.append(line)

    body_text = "\n".join(cleaned_body).strip()

    formatted = f"# 🍽 {title}\n\n"

    if description:
        formatted += f"*{description}*\n\n"

    formatted += body_text

    return formatted

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

st.session_state.temperature_value = 4
st.session_state.default_days = 1
st.session_state.default_hours = 0

col1, col2, col3, col4  = st.columns([1,1,1,1], vertical_alignment="center")

with col1:
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    input_method = st.radio(
        "Food identification method:",
        ["Select Food Category", "Upload Image"]
    )

    if input_method == "Upload Image":

        if st.session_state.uploaded_file is None:
            st.markdown("""
                <style>
                [data-testid='stFileUploaderDropzone'] {
                    background-color: #161514;
                    color: #ffffff;
                }
                </style>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Upload food image",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded is not None:
                st.session_state.uploaded_file = uploaded
                st.rerun()

        else:
            img = Image.open(st.session_state.uploaded_file)
            st.image(img, width=150)

            # Always show this button
            if st.button("🔄 Upload another image", key="upload_another_image"):
                st.rerun()

            # Analyze only once per uploaded image
            with st.spinner("Analyzing food with AI..."):
                processed = preprocess_food_image(img)
                preds = food_model(processed, training=False).numpy()

                if np.isnan(preds).any():
                    st.error("Prediction produced NaN. Check preprocessing or model.")
                    food = "pork"
                else:
                    top_index = np.argmax(preds[0])
                    food_img = FOOD101_LABELS[top_index]
                    food = map_food_category(food_img)
                    confidence = preds[0][top_index]
                    st.success(f"Detected dish: {food_img}")


    else:
        st.markdown("""
        <style>
        .stSelectbox div[data-baseweb="select"] > div:first-child {
            background-color: #161514;
        }
        </style>
        """, unsafe_allow_html=True)

        food = st.selectbox(
            "Select food type:",
            list(FOOD_MODELS.keys()),
            width=200
        ).lower()


with col2:
    temperature = st.slider(
        "Storage Temperature",
        min_value=0,
        max_value=30,
        value=4,
        key="temperature_slider"
    )

    # st.write("Selected temperature:", temperature, "°C")

with col3:
    time_hours = days_hours_input(
        label="Storage Time",
        default_days=st.session_state.default_days,
        default_hours=st.session_state.default_hours,
        key="storage_time"
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
            final_concentration = results.get('final_logC')
            cooking_reco = results.get('cooking_reco')
            times = results.get('times')
            predictions = results.get('logCs')
            fig = results.get('fig')
            time_to_danger_per_bacteria = results.get('time_to_danger_per_bacteria')

            # Store everything needed for display + AI explanation
            st.session_state.prediction_done = True
            st.session_state.prediction = {
                "food": food,
                "temperature": temperature,
                "time_hours": time_hours,
                "is_safe": is_safe,
                "bacterias": bacterias,
                "final_concentration": final_concentration,
                "cooking_reco": cooking_reco,
                "times": times,
                "predictions": predictions,
                "fig": fig,
                "time_to_danger_per_bacteria": time_to_danger_per_bacteria
            }

st.markdown("---")
if st.session_state.prediction_done:
    p = st.session_state.prediction

    if p['is_safe'] == True :
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
    # pathogen_counts = {}
    # for name, pm in PATHOGEN_MODELS.items():
    #     mu_p = ratkowsky_growth_rate(temperature, pm["Tmin"], pm["b"])
    #     N_p = logistic_growth(pm["N0"], pm["Nmax"], mu_p, time_hours)
    #     pathogen_counts[name] = (N_p, mu_p)
    pathogen_counts = p['final_concentration']

    # Total count: simple conservative choice = max of pathogens (or sum if you prefer)
    #total_count = max(v[0] for v in pathogen_counts.values())

    # --- Gauges in 4 columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.plotly_chart(make_gauge("E. coli", pathogen_counts["ec"], "ec"))
        if p["fig"] is not None:
            st.plotly_chart(p["fig"]["ec"])
        else:
            st.warning("No figure returned by infer().")
    with c2:
        st.plotly_chart(make_gauge("Listeria", pathogen_counts["lm"], "lm"))
        if p["fig"] is not None:
            st.plotly_chart(p["fig"]["lm"])
        else:
            st.warning("No figure returned by infer().")
    with c3:
        st.plotly_chart(make_gauge("Salmonella", pathogen_counts["ss"], "ss"))
        if p["fig"] is not None:
            st.plotly_chart(p["fig"]["ss"])
        else:
            st.warning("No figure returned by infer().")
    with c4:
        st.plotly_chart(make_gauge("Total Count", pathogen_counts["ta"], "ta"))
        if p["fig"] is not None:
            st.plotly_chart(p["fig"]["ta"])
        else:
            st.warning("No figure returned by infer().")

    # -----------------------------
    # Remaining shelf-life section
    # -----------------------------
    st.markdown("## ⏳ Remaining Shelf-Life")

    times = p["time_to_danger_per_bacteria"]

    valid_times = {
        k: v for k, v in times.items()
        if v is not None and isinstance(v, (int, float, np.integer, np.floating))
    }

    if valid_times:
        max_risk = min(valid_times, key=valid_times.get)   # bacteria that reaches danger first
        danger_time = valid_times[max_risk]
        elapsed_time = st.session_state.prediction["time_hours"]
        remaining_risk = max(0, float(danger_time) - float(elapsed_time))
    else:
        max_risk = None
        remaining_risk = None

    # Display
    g1, g2 = st.columns([2, 1], vertical_alignment="center")

    with g1:
        st.plotly_chart(
            make_shelflife_gauge(remaining_risk, max_display_hours=72)
        )

    with g2:
        if remaining_risk is not None:
            st.metric("Remaining shelf-life", f"{remaining_risk:.1f} h")
            st.write(f"**Limiting microorganism:** {max_risk}")
        else:
            st.metric("Remaining time until High Risk", "No risk")
            st.write("No microorganism reaches dangerous level in the prediction horizon.")

        if remaining_risk is None:
            st.info("✅ No high-risk threshold reached in the prediction window.")
        elif remaining_risk > 24:
            st.success("✅ Shelf-life is still comfortable.")
        elif remaining_risk > 6:
            st.warning("⚠️ Shelf-life is getting shorter. Use soon.")
        else:
            st.error("❌ Very little safe time remains.")

    # -----------------------------
    # Bacterial growth chart
    # -----------------------------
    if status != "✅ Safe":
        if st.button("What does it mean?"):
            with st.spinner("Generating detailed explanation..."):
                explanations = explanations.risk_explanation(p["bacterias"], max_output_tokens=2000)

            st.markdown("### 🧠 AI Detailed Explanation")
            st.write(explanations)
    else:
        st.markdown("## Recipe suggestions")
        if p["cooking_reco"] == "raw" and p['food'] == "poultry":
            cooking_choice = ["Quick cooking", "High temperature cooking"]
        elif p["cooking_reco"] == "raw":
            cooking_choice = ["Tartare", "Quick cooking", "High temperature cooking"]
        elif p["cooking_reco"] == "medium":
            cooking_choice = ["Quick cooking", "High temperature cooking"]
        else:
            cooking_choice = ["High temperature cooking"]

        cooking_dict = {'Tartare': 'raw',
                    'Quick cooking': 'medium',
                    'High temperature cooking': 'high'}

        recipe_cooking = cooking_dict[st.selectbox('What kind of recipes would you like?', options=cooking_choice, width=300)]
        if recipe_cooking == None:
            st.markdown("Please make a choice")
        if st.button("Find recipes"):
            with st.spinner("Looking for yummy recipes"):
                recipe = recipes.recipe_suggestion(ingredient=p["food"], cooking=recipe_cooking, provider='auto', max_output_tokens=5000)

            if recipe is not None:
                recipe_text = recipe["recipe_text"]
                recipe_image = recipe["image"]
                formatted_recipe = format_recipe_text(recipe_text)
                recipe_title = recipe["recipe_title"]
                short_description = recipe["short_description"]
                key_ingredients = recipe["key_ingredients"]
                cooking_method = recipe["cooking_method"]
                basic_preparation_steps = recipe["basic_preparation_steps"]


    #    "recipe_title": text_bundle["recipe_title"],
    #     "short_description": text_bundle["short_description"],
    #     "key_ingredients": text_bundle["key_ingredients"],
    #     "cooking_method": text_bundle["cooking_method"],
    #     "basic_preparation_steps": text_bundle["basic_preparation_steps"],
    #     "recipe_text": text_bundle["recipe_text"],
    #     "image": image_bundle["image"],
    #     "debug": {
    #         "provider": resolved_provider,
    #         "text_model": model,
    #         "image_model": image_model,
    #         "text_finish_reason": text_bundle.get("__finish_reason") or None,
    #         "image_finish_reason": image_bundle.get("finish_reason"),
    #         "text_attempts": text_bundle.get("__attempts", 1),
                st.markdown(f"## {recipe_title}")
                st.markdown(short_description)
                col1, col2 = st.columns([1, 2], vertical_alignment="top")

                with col1:
                    st.image(recipe_image)

                with col2:
                    st.markdown("### Cooking method")
                    st.markdown(cooking_method)

                    st.markdown("### Ingredients")
                    st.markdown(key_ingredients)

                    st.markdown("### Preparation")
                    st.markdown(basic_preparation_steps)
