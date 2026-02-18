"""
Docstring for app
SaferFood front-end
"""

import streamlit as st
import matplotlib.pyplot as plt
from interface.inference import infer

st.set_page_config(page_title="SaferFood", layout="centered")

st.title("SaferFood", text_alignment='center')

# Food categories to display/select from
FOOD_CATEGORIES = ['beef', 'produce', 'seafood', 'poultry', 'pork']

# Storage temperatures to select from
# STORAGE_TEMP_CATEGORIES = {
#     'Frozen': -18,
#     'Chilled': 4,
#     'Ambiant': 21
# }

#Storage temperatures span and step (°C)
STORAGE_TEMPS_SPAN = {
    'min': 0,
    'max': 30
}
STORAGE_TEMPS_STEP = 5

# Storage time span (minimum and maximum storage time allowed)
# STORAGE_TIME_SPAN = {
#     'min': 1,
#     'max': 1_000
# }

#Storage times
STORAGE_TIMES = {}
# Hours
for h in range(0, 24):
    STORAGE_TIMES[f"{h} h"] = h
# Days
for d in range(1, 22):
    STORAGE_TIMES[f"{d} d"] = d * 24


tab_prediction, tab_explanations, tab_recipes = st.tabs(
    ["Prediction", "Explanations", "Recipes"]
)


with tab_prediction:

    # Create vertical layout (left / right)
    col_left, col_right = st.columns([1, 2])  # ratio adjustable

    # =========================
    # LEFT COLUMN (INPUTS)
    # =========================
    with col_left:

        matrixID = st.selectbox(
            "What do you plan on eating today ?",
            FOOD_CATEGORIES
        )

        temperature_value = st.slider(
            "What was your food's storage temperature ?",
            min_value=STORAGE_TEMPS_SPAN['min'],
            max_value=STORAGE_TEMPS_SPAN['max'],
            step=STORAGE_TEMPS_STEP,
            value=5         #default value : 5°C
        )

        # time = st.slider(
        #     "How long was your food stored (in hours) ?",
        #     min_value=STORAGE_TIME_SPAN['min'],
        #     max_value=STORAGE_TIME_SPAN['max']
        # )

        time_label = st.select_slider(
            "How long was your food stored?",
            options=list(STORAGE_TIMES.keys()),
            value="24 h" if "24 h" in STORAGE_TIMES else "1 d"
        )

        time_in_hours = STORAGE_TIMES[time_label]


        go_for_prediction = st.button("CHECK MY FOOD !", use_container_width=True)

    # =========================
    # RIGHT COLUMN (RESULTS)
    # =========================
    with col_right:

        if go_for_prediction:

            params = {
                "matrixID": matrixID,
                "temperature": temperature_value,
                "time": time_in_hours
            }

            results = infer(params=params)

            if results.get("is_safe") is True:
                st.markdown("# YOUR FOOD IS SAFE TO EAT !")
            else:
                st.markdown("# YOUR FOOD IS UNSAFE TO EAT !")

            fig = results.get("fig")
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("No figure returned by infer().")


with tab_explanations:
    st.header("Explanations")
    st.markdown(
        """
- Explain here how the prediction is computed
- What "safe" means (assumptions, thresholds, limits)
- Model caveats and proper food safety guidance
"""
    )

with tab_recipes:
    st.header("Recipes")
    st.markdown(
        """
- Put recipe ideas here
- Could be conditional on selected food category
- Or show safe-handling / cooking tips
"""
    )
