import numpy as np
import pandas as pd
import psycopg2
import os
import random
import datetime
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import magec_utils as mg
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import class_weight


vitals = ['heartrate_mean', 'sysbp_mean', 'diasbp_mean', 'meanbp_mean',
          'resprate_mean', 'tempc_mean', 'spo2_mean', 'glucose_mean']
labs = ['aniongap', 'albumin', 'bicarbonate', 'bilirubin', 'creatinine',
        'chloride', 'glucose', 'hemoglobin', 'lactate',
        'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 'inr',
        'pt', 'sodium', 'bun', 'wbc']  # -hematocrit
comobs = ['congestive_heart_failure', 'chronic_pulmonary', 'pulmonary_circulation']
others = ['age', 'gender']


def get_mimic_data():
    sqluser = 'postgres'
    dbname = 'mimic'
    schema_name = 'mimiciii'
    engine = create_engine("postgresql+psycopg2://{}:{}@/{}".format(sqluser, sqluser, dbname))
    conn = engine.connect()
    conn.execute('SET search_path to ' + schema_name)
    df = pd.read_sql("SELECT * FROM mimic_users_study;", conn)
    conn.close()
    df = df.sort_values(['subject_id', 'timepoint'], ascending=(1, 0))
    return df


def get_ml_data(df):
    df_ml = df.set_index(['subject_id', 'timepoint']).groupby(level=0, group_keys=False). \
        apply(featurize).reset_index()
    return df_ml


def get_ml_series_data(df):
    df_time = df.set_index(['subject_id']).groupby(level=0, group_keys=False). \
        apply(featurize_time).apply(pd.Series.explode).reset_index()
    return df_time


def train_valid(df_ml):

    def impute(df):
        df[vitals + labs] = df[vitals + labs].fillna(df[vitals + labs].mean())
        df[comobs] = df[comobs].fillna(0)
        return df

    seed = 7
    np.random.seed(seed)

    x = df_ml[list(set(df_ml.columns) - {'subject_id', 'label'})]
    Y = df_ml[['subject_id', 'label']]

    x_train, x_validation, Y_train, Y_validation = train_test_split(x.copy(), Y, test_size=0.2, random_state=seed)
    x_train = impute(x_train)
    x_validation = impute(x_validation)
    
    stsc = StandardScaler()
    xst_train = stsc.fit_transform(x_train)
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=x_train.columns)

    xst_validation = stsc.transform(x_validation)
    xst_validation = pd.DataFrame(xst_validation, index=x_validation.index, columns=x_validation.columns)


def last_val(x):
    vals = x[~np.isnan(x)]
    if len(vals):
        return vals[-1]
    else:
        return None


def featurize_time(df):
    out = dict()
    for i in range(len(df)):
        for lab in labs:
            val = last_val(df[lab].values[:i + 1])
            if lab not in out:
                out[lab] = [val]
            else:
                out[lab].append(val)
        for vital in vitals:
            val = last_val(df[vital].values[:i + 1])
            if vital not in out:
                out[vital] = [val]
            else:
                out[vital].append(val)
        for comob in comobs:
            val = last_val(df[comob].values[:i + 1])
            if comob not in out:
                out[comob] = [val]
            else:
                out[comob].append(val)
        for other in others:
            val = last_val(df[other].values[:i + 1])
            if other not in out:
                out[other] = [val]
            else:
                out[other].append(val)
        out['timepoint'] = df.timepoint.values
        out['label'] = [int(x) for x in df.ventilated.values]
    return pd.Series(out)


def featurize(df):
    out = dict()
    for lab in labs:
        out[lab] = last_val(df[lab])
    for vital in vitals:
        out[vital] = last_val(df[vital])
    for comob in comobs:
        out[comob] = last_val(df[comob])
    for other in others:
        out[other] = last_val(df[other])
    out['label'] = int(df.ventilated.iloc[-1])
    return pd.Series(out)


