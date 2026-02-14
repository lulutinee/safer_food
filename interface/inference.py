'''
Docstring for interface.inference
Effectue l'inférence : reçoit les données entrées par l'utilisateur, et renvoie SAFE/UNSAFE, données, graphe
'''

import numpy as np
import matplotlib.pyplot as plt


#TODO : travailler le type hinting
def infer(params):
    #Valeurs par défaut
    is_safe = None
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
    logCs = np.sin(times/(2*np.pi))
    # print(f'{times=}, {type(times)=}, {type(logCs)=}')

    #Tracé figure TODO : mettre le tracé des courbes dans une fonction à part
    fig, ax = plt.subplots()
    ax.plot(times, logCs)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("logC")
    ax.set_title("Predicted bacterial growth")
    ax.grid(True)

    result = {
        'is_safe': is_safe,
        'times': times,
        'logCs': logCs,
        'fig': fig,
        'ax': ax
    }

    return result
