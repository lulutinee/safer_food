"""
Docstring for app
SaferFood front-end
"""

import streamlit as st
import matplotlib.pyplot as plt
from interface.inference import infer
from interface import explanations
from interface import recipes

st.set_page_config(page_title="SaferFood", layout="centered")

st.title("Is my food safe to eat?", text_alignment='center')
st.markdown('Customer edition', text_alignment='center')

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


# Default values (will be infered from user input)
is_safe = None
bacterias = None
final_logC = None
cooking_reco = None
times = None
logCs = None
fig = None
ax = None

tab_prediction, tab_explanations, tab_recipes = st.tabs(
    ["Prediction", "Explanations", "Recipes"]
)

with tab_prediction:

    # Create vertical layout (left / right)
    col_left, col_right = st.columns([5,11])  # ratio adjustable

    # =========================
    # LEFT COLUMN (INPUTS)
    # =========================
    with col_left:

        #matrixID = st.selectbox(
        #    "What do you plan on eating today ?",
        #    FOOD_CATEGORIES
        #)

        if 'food' not in  st.session_state:
            st.session_state['food'] = 'Make a choice'

        def change_food(food):
            st.session_state['food'] = food

        matrixID = st.session_state['food']
        st.text(matrixID)

        col1, col2, col3, col4  = st.columns(4)

        with col1:
            st.button(':cow2:', on_click=change_food, args=['beef'])
        with col2:
            st.button(':pig:', on_click=change_food, args=['pork'])
        with col3:
            st.button(':rooster:', on_click=change_food, args=['poultry'])
        with col4:
            st.button(':lobster:', on_click=change_food, args=['seafood'])

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
            is_safe = results.get('is_safe')
            bacterias = results.get('bacterias')
            cooking_reco = results.get('cooking_reco')
            fig = results.get('fig')

            if is_safe is True:
                st.markdown("# YOUR FOOD IS SAFE TO EAT !")
            else:
                st.markdown("# YOUR FOOD IS UNSAFE TO EAT !")
                st.markdown('There is a risk of:')
                st.markdown("\n".join(f"- {bacteria}" for bacteria in bacterias))

            if cooking_reco is not None:
                st.markdown(f'Given our predictions, the recommended cooking temperature for your {matrixID} is: {cooking_reco}')

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
    if bacterias is not None:
        explanations = explanations.risk_explanation(bacterias, max_output_tokens=2000)
        st.markdown(explanations)

with tab_recipes:
    st.header("Recipes")
    st.markdown(
        """
- Put recipe ideas here
- Could be conditional on selected food category
- Or show safe-handling / cooking tips
"""
    )

    if cooking_reco is not None:
        st.markdown(f'Given our predictions, the recommended cooking temperature for your {matrixID} is: {cooking_reco}')
        recipes = recipes.recipe_suggestion(ingredient=matrixID, cooking=cooking_reco, provider='auto', max_output_tokens=5000)
        st.markdown(recipes)
