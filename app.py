import streamlit as st
import requests
'''
# SaferFood Front
'''

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
In_on = st.selectbox('What kinf of food?', [
    'ground beef salad with mayonnaise', 'ground roast beef slurry',
    'broth', 'ground beef', 'cod',
    'infant formula hydrated with water',
    'infant formula hydrated with milk',
    'infant formula hydratised with apple juice.', 'BHIB',
    'mash carrot', 'milk', 'minced chicken', 'semi-skimmed milk',
    'chicken liver pate', 'BHI broth', 'liquid egg whites',
    'broccoli puree', 'mushroom puree', 'potato puree',
    'potato puree containing cysteine', 'TGYB', 'beef gravy',
    'raw milk', 'yolk', 'MRSB', 'ham', 'water', 'beef', 'TPB',
    'uht milk', 'all beef dog food', 'canned corn', 'chicken roll',
    'pasteurised crabmeat', 'model frankfurter',
    'Chicken nugget meat blend', 'fish', 'scottish pencake',
    'madeira cake', 'non-dairy cream', 'BHI', 'chicken broth',
    'apple juice', 'whole milk', 'skimmed milk', 'peptone solution',
    'quality burger', 'beefburger', 'quality beefbuger',
    'economy beefbuger', 'beef jerky', 'marinaded beef jerky',
    'beef muscle', 'model aqueous systems based on BHIB',
    'broccoli juice', 'potato juice', 'sweetcorn kernals (canned)',
    'cauliflower puree', 'fresh endive', 'ground turkey',
    'chicken breast', 'nutrient broth', 'Nutrient broth', 'TSB',
    'half cream', 'Double cream', 'Butter', 'sausage meat mixture',
    'fermented meat model medium', 'cooked turkey breast',
    'cooked ham', 'cooked chicken breast', 'watermelon',
    'sprouting alfalfa seeds', 'cottage cheese', 'raw chicken wings'])
temperature = st.selectbox('What is the storage temperature in celcius?', [-18, 4, 20])
time = st.number_input('Insert the number of hours between 1 and 1000')
'''
## Once we have these, let's call our API in order to retrieve a prediction

See ? No need to load a `model.joblib` file in this app, we do not even need to know anything about Data Science in order to retrieve a prediction...

🤔 How could we call our API ? Off course... The `requests` package 💡
'''

#url = 'Add url here'

#params = {"In_on": In_on,
#          "temperature": temperature,
#          "time": time}

#response = round(requests.get(url, params=params).json()['fare'],2)

#st.markdown(f'The predicted fare is: {response} $')
'''

2. Let's build a dictionary containing the parameters for our API...

3. Let's call our API using the `requests` package...

4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
'''
