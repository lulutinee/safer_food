import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Sequence
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from scipy.interpolate import interp1d
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from scipy.interpolate import PchipInterpolator
from keras.layers import Masking
from keras import models, layers, Input, optimizers, metrics, Sequential

def import_data():
    conditions = pd.read_csv('../raw_data/Conditions.txt', sep='\t')
    experiment_conditions = pd.read_csv('../raw_data/ExperimentConditions.txt', sep='\t')
    experiments = pd.read_csv('../raw_data/Experiments.txt', sep='\t', encoding='latin1')
    logcs = pd.read_csv('../raw_data/LogCs.txt', sep='\t')
    properties = pd.read_csv('../raw_data/Properties.txt', sep='\t')
    response_properties = pd.read_csv('../raw_data/ResponseProperties.txt', sep='\t')
    responses = pd.read_csv('../raw_data/Responses.txt', sep='\t', encoding='latin1')
    source = pd.read_csv('../raw_data/Source.txt', sep='\t')

    return conditions, experiment_conditions, experiments, logcs, properties, response_properties, responses, source

def merge_data(conditions, experiment_conditions, experiments, logcs, properties, response_properties, responses, source):
    # Pivot ExperimentConditions table to have one row per ExperimentID before the merge
    # This table is not needed for the actual project, but will be useful for us
    # in the future...
    # experiment_conditions_pivot = experiment_conditions.pivot_table(index='ExperimentID', columns='ConditionID', values='Value').reset_index()

    # Merge the tables to create a single DataFrame for analysis. Will join data based on the logcs table
    data = responses.merge(experiments, left_on='ExperimentID', right_on='ID',
                           how='left'
                           ).drop(columns=['ID_y']
                                  ).rename(columns={'ID_x': 'ResponseID'})
    # data = data.merge(experiment_conditions_pivot, on='ExperimentID', how='left')
    # data = data.merge(response_properties, on='ResponseID', how='left'
    #                  ).rename(columns={'Value_x': 'ResponseValue',
    #                                    'Value_y': 'ResponsePropertiesValue'})
    data = data.merge(logcs, on='ResponseID', how='right')

    return data

def clean_data(df, min_time_threshold=10, max_time_threshold=504, min_temp_threshold=0, max_temp_threshold=30):
    # Keep ResponseIDs with at least 10 time points and a maximum time of 504 hours (21 days)
    response_counts = df.groupby('ResponseID').size()
    valid_response_ids = response_counts[response_counts >= 10].index
    df = df[df['ResponseID'].isin(valid_response_ids)]

    # Remove data points where the maximum time exceeds a certain treshold
    df = df[(df['TObs'] >= min_time_threshold) & (df['TObs'] <= max_time_threshold)]

    # Remove data points where the temperature is outside a certain treshold
    df = df[(df['Temperature'] >= min_temp_threshold) & (df['Temperature'] <= max_temp_threshold)]

    # Keep data from specific matrices only
    food_matrices = ['beef', 'poultry', 'seafood', 'pork']
    df = df[df['MatrixID'].isin(food_matrices)]

    # Keep data from specific organisms only
    organisms = ['ec', 'lm', 'ss']
    df = df[df['OrganismID'].isin(organisms)].reset_index()

    # Fill missing values in the DataFrame based on the provided fill_values
    # dictionary
    fill_values = {
        #'n2': 78.1,         # Must calculate the O2 and CO2 value too...
        'Temperature': 21,
        #'pH':,              # Use mean value from the in_on group
        #'Aw':,              # Use mean value from the in_on group
        #'pressure': 0.101325,
        #'acetic_acid':0,
        #'alta':0,
        #'apple_polyphenol':0,
        #'ascorbic_acid':0,
        #'benzoic_acid':0,
        #'betaine':0,
        #'calcium_propionate':0,
        #'carvacrol':0,
        #'chitosan':0,
        #'cinnamaldehyde':0,
        #'citric_acid':0,
        #'clo2':0,
        #'co2':0.04,         # Must calculate the O2 and N2 value too...
        #'dextrose':0,
        #'diacetic_acid':0,
        #'edta':0,
        #'erythorbate':0,
        #'ethanol':0,
        #'fat':,             # Use mean value from the in_on group or drop
        #'fructose':0,
        #'garlic':0,
        #'glucose':0,
        #'glycerol':0,
        #'green_tea_leaf':0,
        #'green_tea_polyphenol':0,
        #'hcl':0,
        #'irradiated':0,
        #'irradiation':0,
        #'kcl':0,
        #'lactic_acid':0,
        #'lauricidin':0,
        #'malic_acid':0,
        #'moisture':,        # Use mean value from the in_on group
        #'nacl':,            # Use mean value from the in_on group
        #'nitrate':0,
        #'nitrite':0,
        #'o2':21,              # Must calculate the N2 and CO2 value too...
        #'oregano':0,
        #'pomegranate':0,
        #'potassium_lactate':0,
        #'potassium_sorbate':0,
        #'propionic_acid':0,
        #'propylene_oxide':0,
        #'protein':,         # Use mean value from the in_on group or drop
        #'rosemary':0,
        #'sodium_lactate':0,
        #'sorbic_acid':0,
        #'sucrose':0,
        #'sugar':0,
        #'thymol':0
        'PropertyID': 'Other'
    }

    # Impute missing values for 'MethodID' with the most frequent value
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='most_frequent')
    imputer.fit(df[['MethodID']])
    df['MethodID'] = imputer.transform(df[['MethodID']]).ravel()

    # Drop columns that are not needed for the analysis.
    columns_to_drop = ['Spec_rate', 'RateMethod', 'Logc0', 'CombaseID_x',
                       'heated', 'OrganismSpecification', 'Comment', 'Ph', 'Aw',
                       'Value_x', 'ComBaseID', 'ComBaseID_y', 'UserId',
                       'Assumed', 'index', 'ExperimentID', 'LinkId', 'SourceID',
                       'MethodID', 'ID', 'TObs', 'LogcVar']

    for column, value in fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    # Drop columns that are not needed for the analysis
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df

def split_dataset(df):
    # 1. Define the splitter (80% train, 20% test)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    # 2. Split based on Experiment_ID
    # This returns indices for the unique groups
    train_idx, test_idx = next(gss.split(df, groups=df['ResponseID']))

    # 3. Create the actual dataframes
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    return df_train, df_test

def data_engineering(df):
    # Calculate time delta for each ResponseID from the last time point
    df['Time_delta'] = df.groupby('ResponseID')['Time'].diff().fillna(0)

    # Calculate log delta for each ResponseID from the last time point
    df['log_delta'] = df.groupby('ResponseID')['Value'].diff().fillna(0)

    # Calculate time difference between T0 and TObs for each ResponseID
    df['Time_diff'] = df.groupby('ResponseID')['Time'].transform(lambda x: x - x.min()).round(0)

    # Calculate log difference between T0 and TObs for each ResponseID
    df['log_diff'] = df.groupby('ResponseID')['Value'].transform(lambda x: x - x.min())

    df.drop(columns=['In_on'],
            inplace=True)

    indices_to_drop = df[df['Time'] == 0].index
    df.drop(indices_to_drop, inplace=True)

    return df


def interpolate(df):
    master_time_grid = np.arange(0, 504, 1)

    processed_data = []

    for response_id, group in df.groupby('ResponseID'):
        # 1. Ensure data is sorted by time (Crucial for PCHIP)
        group = group.sort_values('Time_diff')

        # 2. Drop duplicates (PCHIP fails if two points have the same time)
        group = group.drop_duplicates(subset=['Time_diff'])

        # 3. Get raw values as NumPy arrays
        x_sparse = group['Time_diff'].values
        y_sparse = group['log_diff'].values

        # Safety Check: We need at least 2 points to interpolate
        if len(x_sparse) < 2:
            continue

        # 4. Create the Interpolator Object
        # PCHIP is "shape-preserving" and won't overshoot your growth data
        interp_func = PchipInterpolator(x_sparse, y_sparse)

        # 5. Define points to calculate (only up to the end of this specific experiment)
        max_time = x_sparse.max()
        valid_grid_points = master_time_grid[master_time_grid <= max_time]

        # 6. Generate the new values
        y_interp = interp_func(valid_grid_points)

        # Rebuild the dataframe for this sequence
        temp_df = pd.DataFrame({
            'ResponseID': response_id,
            'Time_diff': valid_grid_points,
            'log_diff': y_interp,
            'Temperature': group['Temperature'].iloc[0], # Static feature
            'MatrixID': group['MatrixID'].iloc[0],  # Static feature
            'OrganismID': group['OrganismID'].iloc[0]   # Static feature
        })

        processed_data.append(temp_df)

    final_df = pd.concat(processed_data, ignore_index=True)

    return final_df

def pad_data(df):
    # We calculate max per group for Y
    y_values = df.groupby('ResponseID')['log_diff'].max().values.reshape(-1, 1)

    # 3. Building X Sequences
    X_list = []
    # It's vital to iterate through the groups in the same order as we did for Y
    for x_id in df['ResponseID'].unique():
        group = df[df['ResponseID'] == x_id]
        # Features: Time, Temp, and the encoded Food Matrix
        features = group[['Time_diff', 'Temperature', 'MatrixID_beef',
                        'MatrixID_pork', 'MatrixID_poultry', 'MatrixID_seafood',
                        'OrganismID_ec', 'OrganismID_lm', 'OrganismID_ss']].values
        X_list.append(features)

    # 4. Padding X
    X_padded = pad_sequences(X_list, padding='post', dtype='float32', value=1000)

    return X_padded, y_values

def initialize_model():
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:])) # input shape is (input_length, n_features)
    model.add(layers.Masking(mask_value=1000))
    model.add(layers.LSTM(64,return_sequences=False))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    return model

def compile_model(model):
    model.compile(loss='mae',
                optimizer='adam')
    return model
