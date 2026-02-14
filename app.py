import streamlit as st
import requests
import matplotlib.pyplot as plt
from interface.inference import infer
'''
# SaferFood Front
'''

#Food categories to display/select from
FOOD_CATEGORIES = ['beef', 'produce', 'seafood', 'poultry', 'pork']
#Storage temperatures to select from
STORAGE_TEMP_CATEGORIES = {
    'Frozen': -18,
    'Chilled': 4,
    'Ambiant': 21
}
#Storage time span (minimum and maximum storage time allowed)
STORAGE_TIME_SPAN = {
    'min': 1,
    'max': 1_000
    }


#INPUTS

st.markdown('''
Is it good to eat?
''')

'''
## Input parameters

1. Let's ask for:
- Kind of food
- Temperature
- Timelength of the storage
'''

#Input food
matrixID = st.selectbox(
    'What do you plan on eating today ?',
    FOOD_CATEGORIES)
#Input temperature
temperature = st.selectbox(
    'How was your food stored',
    STORAGE_TEMP_CATEGORIES.keys()
    )
#Input time
time = st.slider(
    'How long was your food stored (in hours) ?',
    min_value = STORAGE_TIME_SPAN['min'],
    max_value = STORAGE_TIME_SPAN['max']
    )

#Inputs all completed, press button to predict
go_for_prediction = st.button(
    'CHECK MY FOOD !'
    )

#Wrap inputs
params = {
    'matrixID': matrixID,
    'temperature': temperature,
    'time': time
    }

if go_for_prediction:
        st.write(
            f"So you want to know if your {matrixID} "
            f"that you kept at {temperature}°C "
            f"for {time} hours is safe to eat?"
        )


#Predict

results = infer(params=params)

if results['is_safe']:
    st.markdown('''
                # YOUR FOOD IS SAFE TO EAT !
                ''')
elif results['is_safe'] == False:
    st.markdown('''
                # YOUR FOOD IS UNSAFE TO EAT !
                ''')

st.markdown('''
            Predicted bacterial growth
            ''')
st.pyplot(results['fig'])
