import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
    data = data.merge(response_properties, on='ResponseID', how='left'
                      ).rename(columns={'Value_x': 'ResponseValue',
                                        'Value_y': 'ResponsePropertiesValue'})
    data = data.merge(logcs, on='ResponseID', how='right')

    return data

def select_dataset(df, min_time_threshold=5, max_time_threshold=504):
    # Remove data points where the maximum time exceeds a certain treshold
    filtered_df = df[(df['TObs'] >= min_time_threshold) & (df['TObs'] <= max_time_threshold)]

    return filtered_df

def clean_data(df, min_time_threshold=5, max_time_threshold=504, min_temp_threshold=0, max_temp_threshold=30):
    # Remove data points where the maximum time exceeds a certain treshold
    df = df[(df['TObs'] >= min_time_threshold) & (df['TObs'] <= max_time_threshold)]

    # Remove data points where the temperature is outside a certain treshold
    df = df[(df['Temperature'] >= min_temp_threshold) & (df['Temperature'] <= max_temp_threshold)]

    # Keep data from specific matrices only
    food_matrices = ['beef', 'poultry', 'produce', 'seafood', 'pork']
    df = df[df['MatrixID'].isin(food_matrices)].reset_index()

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
                       'Value_x', 'ComBaseID_x', 'ComBaseID_y', 'UserId',
                       'Assumed', 'index', 'ExperimentID', 'LinkId', 'SourceID',
                       'MethodID']

    for column, value in fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    # Drop columns that are not needed for the analysis
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df


