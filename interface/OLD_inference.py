'''
Docstring for interface.inference
Effectue l'inférence : reçoit les données entrées par l'utilisateur, et renvoie SAFE/UNSAFE, seuil de cuisson, données, graphe
Paramètres à fournir :
            params = {
                "matrixID": matrixID,
                "temperature": temperature_value,
                "time": time_in_hours
            }
'''

import numpy as np
import matplotlib.pyplot as plt
#from ml_logic.baseline import ClassicalModel
from ml_logic.sk_baseline import ClassicalModelRegressor
from interface.bacteria_information import MICROORGANISM
from . import arrhenius_parameters #arrhenius_parameters.csv was loaded in the arrhenius_parameters dataframe in interface/__init__.py (just once)
from interface.graphics import plot_predictions_over_time


MAX_PREDICTION_TIME = 1000 # Maximum prediction time (to compute the max. storage time)

#TODO : travailler le type hinting
def infer(params):

    # Paramètres fournis:
    # params = {
    #     "matrixID": matrixID,
    #     "temperature": temperature_value,
    #     "time": time_in_hours
    # }

    # Valeurs retournées par défaut
    is_safe = None
    bacterias = None
    final_logC = None
    cooking_reco = None
    times = None
    logCs = None
    fig = None
    ax = None


    #TODO vérification des données

    # TODO Pour l'instant : nodèle classique, remplacer par ML/AI

    # primary_model = ClassicalModelRegressor('baranyi')
    # secondary_model = ClassicalModelRegressor('arrhenius')

    # liste de temps pour la représentation graphique
    times = np.linspace(
        start = 0,
        stop=params['time'],
        num=100
        )

    # Prédictions
    # 1) récupérer les paramètres secondaires (par OrganismID)
    organism_secondary_params = get_arrhenius_params(params, arrhenius_parameters)
    # 2) prédire
    predictions = predict_for_all_organisms(params, organism_secondary_params)

    # Extract final growth and compute final concentration from growth predictions for every bacteria
    final_growth = {}
    for bacteria, values in predictions.items():
        if values is None or len(values) == 0:
            raise ValueError(f"No prediction values for bacteria '{bacteria}'")
        final_growth[bacteria] = float(values[-1])

    final_concentration = {}
    for bacteria, growth_value in final_growth.items():
        if bacteria not in MICROORGANISM:
            raise ValueError(f"Bacteria '{bacteria}' not found in MICROORGANISM dictionary")
        if "initial" not in MICROORGANISM[bacteria]:
            raise ValueError(f"'initial' value missing for bacteria '{bacteria}'")

        initial_value = MICROORGANISM[bacteria]["initial"]
        final_concentration[bacteria] = float(growth_value + initial_value)

    # Check if the food is safe, and what bacterias are present at significant concentrations (>= 'raw' threshold)

    # Default conservative assumption
    is_safe = False
    all_below_high = True
    bacterias = []

    for bacteria, concentration in final_concentration.items():
        if bacteria not in MICROORGANISM:
            raise ValueError(f"Bacteria '{bacteria}' not found in MICROORGANISM")
        data = MICROORGANISM[bacteria]

        high_threshold = data.get("high")
        raw_threshold = data.get("raw")
        usual_name = data.get("usual_name")

        if high_threshold is None:
            raise ValueError(f"'high' threshold missing for bacteria '{bacteria}'")
        if raw_threshold is None:
            raise ValueError(f"'raw' threshold missing for bacteria '{bacteria}'")
        if usual_name is None:
            raise ValueError(f"'usual_name' missing for bacteria '{bacteria}'")

        # Safety check (strictly below high threshold required)
        if concentration >= high_threshold:
            all_below_high = False

        # Raw threshold check (list risky bacteria)
        if concentration >= raw_threshold:
            bacterias.append(usual_name)

    # Final safety decision
    if all_below_high:
        is_safe = True



    # From final bacterial load, decide the cooking reconmmendation (but just for )
    ta_values = MICROORGANISM['ta']
    ta_final_concentration = final_concentration.get('ta')

    if ta_final_concentration >= ta_values['high']:
        cooking_reco = 'unsafe'
    elif ta_final_concentration >= ta_values['medium']:
        cooking_reco = 'high'
    elif ta_final_concentration >= ta_values['raw']:
        cooking_reco = 'medium'
    else:
        cooking_reco = 'raw'


    #Tracé figure TODO : mettre le tracé des courbes dans une fonction à part
    fig = plot_predictions_over_time(times, predictions, highlight_bacteria='ta')



    # Package everything in a dictionary and return the data
    result = {
        'is_safe': is_safe,                 # Is the food safe ?
        'bacterias': bacterias,             # What are the bacteria whose final concentration might be a concenr ?
        'final_logC': final_concentration,  # What are the final concentrations of the various bacteria ?
        'cooking_reco': cooking_reco,       # What are the cooking recommendations ?
        'times': times,                     # Predicted growth : storage time
        'logCs': predictions,               # Predicted growth : concentration of each bacteria
        'fig': fig                          # Graphics of growth vs time
    }

    return result


# Fonctions pour la détermination des paramètres du modèle primaire, et la modélisation

def get_arrhenius_params(params: dict, arrhenius_parameters):
    matrix_id = params.get("matrixID")

    if matrix_id is None:
        raise ValueError("Parameter 'matrixID' is missing")

    # Filtrage
    df_filtered = arrhenius_parameters[
        arrhenius_parameters["MatrixID"] == matrix_id
    ]

    # Vérifications
    if df_filtered.empty:
        raise ValueError(f"MatrixID '{matrix_id}' not found in arrhenius_parameters")
    if df_filtered.duplicated(subset=["OrganismID"]).any():
        raise ValueError(f"Duplicate OrganismID entries for MatrixID '{matrix_id}'")

    # Colonnes à récupérer
    cols = ["Initial Value", "Lag", "Maximum Rate", "Final Value"]

    # Construction d’un dictionnaire par OrganismID
    result = (
        df_filtered
        .set_index("OrganismID")[cols]
        .to_dict(orient="index")
    )

    return result

from typing import Any, Dict

def predict_for_all_organisms(
    params: dict,
    organism_secondary_params: Dict[Any, Dict[str, dict]],
):
    """
    params: dict contenant au minimum:
        - "temperature": float (ou array-like compatible)
        - "time": float (ou array-like compatible)
    organism_secondary_params:
        dict[OrganismID] -> dict[param_name] -> dict(params du modèle arrhenius)
        Ex:
        {
          "lm": {
             "Lag": {"a": ..., "b": ...},
             "Maximum Rate": {"a": ..., "b": ...},
             ...
          },
          ...
        }

    Retour:
        predictions: dict[OrganismID] -> prediction du modèle primaire (baranyi) à time
    """
    temperature = params.get("temperature")
    time =     times = np.linspace(
        start = 0,
        stop=params.get('time'),
        num=100
        )

    if temperature is None:
        raise ValueError("Parameter 'temperature' is missing")
    if time is None:
        raise ValueError("Parameter 'time' is missing")

    primary_param_names = ["Initial Value", "Lag", "Maximum Rate", "Final Value"]

    predictions: Dict[Any, Any] = {}

    for organism_id, sec_params_by_primary_param in organism_secondary_params.items():
        # 1) Construire les paramètres du modèle primaire en prédisant chaque paramètre via le secondaire
        primary_params: Dict[str, float] = {}

        for p_name in primary_param_names:
            if p_name not in sec_params_by_primary_param:
                raise ValueError(
                    f"Missing secondary-model parameters for OrganismID={organism_id}, "
                    f"primary parameter='{p_name}'"
                )

            # 2) Modèle secondaire pour ce paramètre p_name
            secondary_model = ClassicalModelRegressor(model="arrhenius")
            secondary_model.fit([1, 2, 3], [1, 2, 3]) #On doit fitter le modèle sinon on a une erreur, même si on force les params

            # On récupère les paramètres arrhenius correspondant à (organism_id, p_name)
            secondary_model.params_ = sec_params_by_primary_param[p_name]

            # 3) On prédit la valeur du paramètre du modèle primaire à la température donnée
            # Important: selon ton implémentation, predict peut renvoyer array([[x]]) ou array([x])
            pred_value = secondary_model.predict([temperature])

            # Normalisation "souple" vers scalaire si besoin
            try:
                # numpy-like
                if hasattr(pred_value, "shape"):
                    pred_value = float(pred_value.ravel()[0])
                else:
                    pred_value = float(pred_value)
            except Exception:
                # si tu veux garder l'objet tel quel
                pass

            primary_params[p_name] = pred_value

        # 4) Modèle primaire (baranyi) avec les paramètres prédits
        primary_model = ClassicalModelRegressor(model="baranyi")
        primary_model.fit([1, 2, 3], [1, 2, 3]) #On doit fitter le modèle sinon, erreur
        primary_model.params_ = primary_params

        # 5) Prediction à time
        y_pred = primary_model.predict(time)

        # Normalisation éventuelle
        # try:
        #     if hasattr(y_pred, "shape"):
        #         y_pred = float(y_pred.ravel()[0])
        #     else:
        #         y_pred = float(y_pred)
        # except Exception:
        #     pass

        predictions[organism_id] = y_pred

    return predictions


# Helper function : from an OrganismID returns the usual_name (human-friendly name) as defined in the constant MICROORGANISM

def get_usual_name(keys):
    """
    Retrieve usual_name(s) from MICROORGANISM.

    Parameters
    ----------
    keys : str or list[str]
        Bacteria code or list of bacteria codes.
    MICROORGANISM : dict
        Dictionary containing bacteria metadata.

    Returns
    -------
    str or list[str]
        Usual name(s) corresponding to the provided key(s).

    Raises
    ------
    ValueError
        If a key does not exist or if 'usual_name' is missing.
    """

    # Normalize to list
    single_input = False
    if isinstance(keys, str):
        keys = [keys]
        single_input = True

    if not isinstance(keys, (list, tuple)):
        raise TypeError("keys must be a string or a list/tuple of strings")

    usual_names = []

    for key in keys:
        if key not in MICROORGANISM:
            raise ValueError(f"Bacteria key '{key}' not found in MICROORGANISM")

        if "usual_name" not in MICROORGANISM[key]:
            raise ValueError(f"'usual_name' missing for bacteria '{key}'")

        usual_names.append(MICROORGANISM[key]["usual_name"])

    return usual_names[0] if single_input else usual_names
