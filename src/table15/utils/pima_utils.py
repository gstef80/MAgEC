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
from . import pipeline_utils as plutils
import os


def pima_data(configs):
    """
    Load PIMA data, impute, scale and train/valid split
    :return:
    """

    def impute(df):
        out = df.copy()
        exclude_cols = plutils.get_from_configs(configs, 'EXCLUDE_COLS')
        cols = list(set(df.columns) - set(exclude_cols))
        out[cols] = out[cols].replace(0, np.NaN)
        out[cols] = out[cols].fillna(out[cols].mean())
        return out

    filename = plutils.get_from_configs(configs, 'DIABS_PATH')
    pima = pd.read_csv(filename)

    random_seed = plutils.get_from_configs(configs, 'RANDOM_SEED', param_type='hyperparams')
    if random_seed is not None:
        np.random.seed(random_seed)
    x = pima.iloc[:, 0:-1]
    Y = pima.iloc[:, -1]
    test_size = plutils.get_from_configs(configs, 'TEST_SIZE', param_type='hyperparams')

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

    return pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p


def create_mlp(x_train_p=None):
    mlp = Sequential()
    mlp.add(Dense(60, input_dim=len(x_train_p.columns), activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(30, input_dim=60, activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1, activation='sigmoid'))
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mlp


def pima_models(x_train_p, y_train_p, models):
    """
    3 ML models for PIMA (scaled) data
    :param x_train_p:
    :param Y_train:
    :return:
    """
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
    
    models_dict = dict()
    for model_name, clf in estimators:
        models_dict[model_name] = clf
    
    # create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft')
    ensemble._estimator_type = "classifier"
    ensemble.fit(x_train_p, y_train_p.values.ravel())
    models_dict['ensemble'] = ensemble
    
    return models_dict


def plot_stats(dfplot, save=False):
    dfplot = dfplot.set_index('Feature')
    dfplot.plot(kind='bar',
                stacked=True,
                figsize=(10, 6),
                title='MAgEC (best) features by model and policy',
                rot=45)
    if save:
        plt.savefig('pima_magec_stats.png', bbox_inches='tight')
    return


def df_stats(stats, con1, con3):

    def feat_num(feat, stats, model):
        if feat in stats and np.any([model in x[1] for x in stats[feat]]):
            return [x[0] for x in stats[feat] if x[1] == model][0]
        else:
            return 0

    dfplot = pd.DataFrame(columns=['Feature', 'LR', 'RF', 'MLP', 'CON@1', 'CON@3'],
                          data=[['Glucose',
                                 feat_num('Glucose', stats, 'lr'),
                                 feat_num('Glucose', stats, 'rf'),
                                 feat_num('Glucose', stats, 'mlp'),
                                 con1['Glucose'][0] if 'Glucose' in con1 else 0,
                                 con3['Glucose'][0] if 'Glucose' in con3 else 0],
                                ['Insulin',
                                 feat_num('Insulin', stats, 'lr'),
                                 feat_num('Insulin', stats, 'rf'),
                                 feat_num('Insulin', stats, 'mlp'),
                                 con1['Insulin'][0] if 'Insulin' in con1 else 0,
                                 con3['Insulin'][0] if 'Insulin' in con3 else 0],
                                ['BMI',
                                 feat_num('BMI', stats, 'lr'),
                                 feat_num('BMI', stats, 'rf'),
                                 feat_num('BMI', stats, 'mlp'),
                                 con1['BMI'][0] if 'BMI' in con1 else 0,
                                 con3['BMI'][0] if 'BMI' in con3 else 0],
                                ['BloodPressure',
                                 feat_num('BloodPressure', stats, 'lr'),
                                 feat_num('BloodPressure', stats, 'rf'),
                                 feat_num('BloodPressure', stats, 'mlp'),
                                 con1['BloodPressure'][0] if 'BloodPressure' in con1 else 0,
                                 con3['BloodPressure'][0] if 'BloodPressure' in con3 else 0],
                                ['SkinThickness',
                                 feat_num('SkinThickness', stats, 'lr'),
                                 feat_num('SkinThickness', stats, 'rf'),
                                 feat_num('SkinThickness', stats, 'mlp'),
                                 con1['SkinThickness'][0] if 'SkinThickness' in con1 else 0,
                                 con3['SkinThickness'][0] if 'SkinThickness' in con3 else 0],
                                ['not_found',
                                 feat_num('not_found', stats, 'lr'),
                                 feat_num('not_found', stats, 'rf'),
                                 feat_num('not_found', stats, 'mlp'),
                                 con1['not_found'][0] if 'not_found' in con1 else 0,
                                 con3['not_found'][0] if 'not_found' in con3 else 0]])
    return dfplot


def plot_pima_features(df):
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax[0, 0] = mg.plot_feature(df, 'BMI', ax[0, 0])
    ax[0, 1] = mg.plot_feature(df, 'BloodPressure', ax[0, 1])
    ax[1, 0] = mg.plot_feature(df, 'DiabetesPedigreeFunction', ax[1, 0])
    ax[1, 1] = mg.plot_feature(df, 'Glucose', ax[1, 1])
    ax[2, 0] = mg.plot_feature(df, 'Insulin', ax[2, 0])
    ax[2, 1] = mg.plot_feature(df, 'SkinThickness', ax[2, 1])
    return ax
