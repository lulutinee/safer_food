'''
Docstring for interface.inference
Effectue l'inférence : reçoit les données entrées par l'utilisateur, et renvoie SAFE/UNSAFE, seuil de cuisson, données, graphe
'''

import numpy as np
import matplotlib.pyplot as plt
from ml_logic.baseline import ClassicalModel
from interface.bacteria_information import MICROORGANISM


#TODO : travailler le type hinting
def infer(params):
    #Valeurs par défaut
    is_safe = None
    bacterias = None
    final_logC = None
    cooking_reco = None
    times = None
    logCs = None
    fig = None
    ax = None


    #TODO vérification des données

    #TODO application du modèle

    #dummy responses, just for MVP
    is_safe = True
    # print(f'{params=}')
    times = np.linspace(
        start = 0,
        stop=params['time'],
        num=100
        )

    # Dummy model, hard-coded for demo
    # TODO replace by real model call
    model = ClassicalModel()
    model.model = 'baranyi'
    model.params = {'Initial Value' : 3.143,
                    'Lag': 33.259,
                    'Maximum Rate':0.0399,
                    'Final Value': 9.0967
                    }
    logCs = model.predict(x = times)
    # print(f'{times=}, {type(times)=}, {type(logCs)=}')

    # Extract final bacterial load
    final_logC = logCs[-1]

    # From final bacterial load, decide if the produc is safe to eat or not
    # Dummy model : let's predict for ta so we have a difference between raw, medium, high

    # Find most problematic pathogen
    bacterias = [
        data['usual_name']
        for data in MICROORGANISM.values()
        if final_logC >= data['raw']
    ]

    if bacterias == []:
        is_safe = True          # Final pathogen load lower than safety threshold
        bacterias = None        # Let's clear the bacteria list
    else:
        is_safe = False



    # From final bacterial load, decide the cooking reconmmendation
    ta_values = MICROORGANISM['ta']

    if final_logC >= ta_values['high']:
        cooking_reco = 'high'
    elif final_logC >= ta_values['medium']:
        cooking_reco = 'medium'
    elif final_logC >= ta_values['raw']:
        cooking_reco = 'raw'
    else:
        cooking_reco = 'raw'


    #Tracé figure TODO : mettre le tracé des courbes dans une fonction à part
    fig, ax = plt.subplots()
    ax.plot(times, logCs)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("logC")
    ax.set_title("Predicted bacterial growth")
    ax.grid(True)

    result = {
        'is_safe': is_safe,
        'bacterias': bacterias,
        'final_logC': final_logC,
        'cooking_reco': cooking_reco,
        'times': times,
        'logCs': logCs,
        'fig': fig,
        'ax': ax
    }

    return result
