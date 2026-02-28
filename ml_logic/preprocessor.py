"""
Docstring for ml_logic.preprocessor
Gère les preprocessors utilisés dans safer_food
    -
    -

"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Sequence
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from scipy.interpolate import interp1d
from keras.preprocessing.sequence import pad_sequences
from scipy.interpolate import PchipInterpolator
from keras.layers import Masking
from keras import models, layers, Input, optimizers, metrics, Sequential

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Pipeline that transforms a cleaned dataset of shape (_, 8)
        into a preprocessed one of fixed shape (_, 11).

        Stateless operation: "fit_transform()" equals "transform()".
        """

        # TIME PIPE
        timedelta_min = 1
        timedelta_max = 504
        time_pipe = FunctionTransformer(lambda timedelta: (timedelta - timedelta_min) / (timedelta_max - timedelta_min))

        # TEMPERATURE PIPE
        temp_min = 0
        temp_max = 30

        temp_pipe = FunctionTransformer(lambda temp: (temp - temp_min) / (temp_max - temp_min))

        # MATRIX PIPE
        matrix_categories = ['Beef', 'Pork', 'Poultry', 'Seafood']


        matrix_pipe = OneHotEncoder(
                    categories=matrix_categories,
                    sparse_output=False,
                    handle_unknown="ignore"
        )

        # ORGANISM PIPE
        organism_categories = ['lm', 'ec', 'ss']


        organism_pipe = OneHotEncoder(
                    categories=organism_categories,
                    sparse_output=False,
                    handle_unknown="ignore"
        )




        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("time_scaler", time_pipe, ["time_diff"]),
                ("temp_scaler", temp_pipe, ["temperature"]),
                ("matrix_preproc", matrix_pipe, matrix_categories),
                ("organism_preproc", organism_pipe, organism_categories),
            ],
            n_jobs=-1,
        )

        return final_preprocessor

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
