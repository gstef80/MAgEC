from typing import Dict
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from . import magec_utils as mg
import os

def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        parsed_yaml = yaml.safe_load(file)
    return parsed_yaml

    
def get_from_configs(configs: Dict, key: str, param_type: str=None):
    key = key.upper()
    if param_type in configs and key in configs[param_type]:
        return configs[param_type][key]
    if key in configs['CONFIGS']:
        return configs['CONFIGS'][key]
    print(f'Warning: could not locate param {key} in configs')
    return None


def generate_data(configs: Dict):
    def impute(df):
        out = df.copy()
        cols = df.columns
        out[cols] = out[cols].replace(0, np.NaN)
        out[cols] = out[cols].fillna(out[cols].mean())
        return out

    csv_path = get_from_configs(configs, 'CSV_PATH')

    numerical_features = get_from_configs(configs, 'NUMERICAL', param_type='FEATURES')
    categorical_features = get_from_configs(configs, 'CATEGORICAL', param_type='FEATURES')
    binary_features = get_from_configs(configs, 'BINARY', param_type='FEATURES')
    target_feature = get_from_configs(configs, 'TARGET', param_type='FEATURES')

    random_seed = get_from_configs(configs, 'RANDOM_SEED', param_type='HYPERPARAMS')
    test_size = get_from_configs(configs, 'TEST_SIZE', param_type='HYPERPARAMS')

    df = pd.read_csv(csv_path)

    if random_seed is not None:
        np.random.seed(random_seed)

    features = numerical_features
    x = df.loc[:, features]
    Y = df.loc[:, target_feature]

    x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=test_size, random_state=random_seed)
    
    x_train = impute(x_train)
    x_validation = impute(x_validation)

    stsc = StandardScaler()
    xst_train = stsc.fit_transform(x_train)
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=x_train.columns)
    xst_validation = stsc.transform(x_validation)
    xst_validation = pd.DataFrame(xst_validation, index=x_validation.index, columns=x_validation.columns)

    # Format
    x_validation_p = xst_validation.copy()
    x_validation_p['timepoint'] = 0
    x_validation_p['case'] = np.arange(len(x_validation_p))
    x_validation_p.set_index(['case', 'timepoint'], inplace=True)
    x_validation_p = x_validation_p.sort_index(axis=1)

    y_validation_p = pd.DataFrame(Y_validation.copy())
    y_validation_p['timepoint'] = 0
    y_validation_p['case'] = np.arange(len(x_validation_p))
    y_validation_p.set_index(['case', 'timepoint'], inplace=True)
    y_validation_p = y_validation_p.sort_index(axis=1)

    # Format
    x_train_p = xst_train.copy()
    x_train_p['timepoint'] = 0
    x_train_p['case'] = np.arange(len(x_train_p))
    x_train_p.set_index(['case', 'timepoint'], inplace=True)
    x_train_p = x_train_p.sort_index(axis=1)

    y_train_p = pd.DataFrame(Y_train.copy())
    y_train_p['timepoint'] = 0
    y_train_p['case'] = np.arange(len(y_train_p))
    y_train_p.set_index(['case', 'timepoint'], inplace=True)
    y_train_p = y_train_p.sort_index(axis=1)

    return df, features, x_train_p, x_validation_p, y_train_p, y_validation_p


def create_mlp(x_train_p=None):
    mlp = Sequential()
    mlp.add(Dense(60, input_dim=len(x_train_p.columns), activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(30, input_dim=60, activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1, activation='sigmoid'))
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mlp


def train_models(x_train_p, y_train_p, models, configs):
    """
    3 ML models for scaled data
    :param x_train_p:
    :param y_train_p:
    :return:
    """
    use_ensemble = get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')

    estimators = list()

    if 'lr' in models:
        lr = LogisticRegression(C=1.)
        lr.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('lr', lr))

    if 'rf' in models:
        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(x_train_p, y_train_p.values.ravel())
        sigmoidRF = CalibratedClassifierCV(RandomForestClassifier(n_estimators=1000), cv=5, method='sigmoid')
        sigmoidRF.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('rf', sigmoidRF))

    if 'mlp' in models:
        params = {'x_train_p': x_train_p}
        mlp = KerasClassifier(build_fn=create_mlp, x_train_p=x_train_p, epochs=100, batch_size=64, verbose=0)
        mlp._estimator_type = "classifier"
        mlp.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('mlp', mlp))
    
    if use_ensemble:
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting='soft')
        ensemble._estimator_type = "classifier"
        ensemble.fit(x_train_p, y_train_p.values.ravel())
        estimators.append(('ensemble', ensemble))
    
    models_dict = dict()
    for model_name, clf in estimators:
        models_dict[model_name] = clf
    
    return models_dict
