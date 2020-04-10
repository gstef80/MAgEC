import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import magec_utils as mg
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


vitals = ['heartrate_mean', 'sysbp_mean', 'diasbp_mean', 'meanbp_mean',
          'resprate_mean', 'tempc_mean', 'spo2_mean', 'glucose_mean']
labs = ['aniongap', 'albumin', 'bicarbonate', 'bilirubin', 'creatinine',
        'chloride', 'glucose', 'hemoglobin', 'lactate',
        'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 'inr',
        'pt', 'sodium', 'bun', 'wbc']  # -hematocrit
comobs = ['congestive_heart_failure', 'chronic_pulmonary', 'pulmonary_circulation']
others = ['age', 'gender']


def impute(df):
    df[vitals + labs] = df[vitals + labs].fillna(df[vitals + labs].mean())
    df[comobs] = df[comobs].fillna(0)
    return df


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


def train_valid_ml(df_ml):
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

    return x_train, x_validation, stsc, xst_train, xst_validation, Y_train, Y_validation


def train_valid_series(df_time, Y_validation):
    # Get train/valid
    train_ind = df_time[~np.isin(df_time['subject_id'], Y_validation.subject_id.unique())].index
    valid_ind = df_time[np.isin(df_time['subject_id'], Y_validation.subject_id.unique())].index

    # Impute
    df_series_train = impute(df_time.iloc[train_ind].copy())
    df_series_valid = impute(df_time.iloc[valid_ind].copy())

    # Get X, Y as numpy arrays
    df_series_train_X = df_series_train[list(set(df_series_train.columns) -
                                             {'subject_id', 'label', 'timepoint'})].astype(float)

    df_series_train_Y = df_series_train[['subject_id', 'label', 'timepoint']]

    df_series_valid_X = df_series_valid[list(set(df_series_valid.columns) -
                                             {'subject_id', 'label', 'timepoint'})].astype(float)

    df_series_valid_Y = df_series_valid[['subject_id', 'label', 'timepoint']]

    # scale
    stsc = StandardScaler()
    tmp = stsc.fit_transform(df_series_train_X)
    df_series_train_X = pd.DataFrame(tmp, index=df_series_train_X.index, columns=df_series_train_X.columns)
    tmp = stsc.transform(df_series_valid_X)
    df_series_valid_X = pd.DataFrame(tmp, index=df_series_valid_X.index, columns=df_series_valid_X.columns)

    # concat X/Y for train/valid
    df_series_train = pd.concat([df_series_train_X, df_series_train_Y], axis=1)
    df_series_valid = pd.concat([df_series_valid_X, df_series_valid_Y], axis=1)
    df_series_train = df_series_train.rename(columns={"subject_id": "case"})
    df_series_valid = df_series_valid.rename(columns={"subject_id": "case"})

    df_series_train = df_series_train.set_index(['case', 'timepoint'])
    df_series_train = df_series_train.sort_index(level=[0, 1], ascending=[1, 0])

    df_series_valid = df_series_valid.set_index(['case', 'timepoint'])
    df_series_valid = df_series_valid.sort_index(level=[0, 1], ascending=[1, 0])

    xt_train, Yt_train, _ = mg.zero_pad(df_series_train)
    xt_valid, Yt_valid, _ = mg.zero_pad(df_series_valid)

    return stsc, df_series_train, df_series_valid, xt_train, Yt_train, xt_valid, Yt_valid


def mimic_models(xst_train, Y_train, xt_train, Yt_train, class_weights):

    def create_mlp():
        mlp = Sequential()
        mlp.add(Dense(60, input_dim=len(xst_train.columns), activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(30, input_dim=60, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(1, activation='sigmoid'))
        mlp.compile(loss='binary_crossentropy',
                    loss_weights=[class_weights[1]], optimizer='adam', metrics=['accuracy'])
        return mlp

    def create_lstm():
        lstm = Sequential()
        lstm.add(LSTM(32, dropout=0.3, recurrent_dropout=0.1, input_shape=xt_train.shape[1:]))
        lstm.add(Dense(1, activation='sigmoid'))
        lstm.compile(loss='binary_crossentropy',
                     loss_weights=[class_weights[1]],
                     optimizer='adam',
                     metrics=['accuracy'])
        return lstm

    mlp = KerasClassifier(build_fn=create_mlp, epochs=100, batch_size=64, verbose=0)
    mlp.fit(xst_train, Y_train['label'], epochs=100, batch_size=64, verbose=0)

    rf = CalibratedClassifierCV(RandomForestClassifier(n_estimators=800,
                                                       min_samples_split=2,
                                                       min_samples_leaf=4,
                                                       max_features='sqrt',
                                                       max_depth=90,
                                                       bootstrap=True,
                                                       n_jobs=-1,
                                                       class_weight="balanced"),
                                method='sigmoid', cv=5)
    rf.fit(xst_train, Y_train['label'])

    lr = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs')
    lr.fit(xst_train, Y_train['label'])

    lstm = KerasClassifier(build_fn=create_lstm, epochs=100, batch_size=64, verbose=0)
    lstm.fit(xt_train, Yt_train, epochs=100, batch_size=64, verbose=0)

    return {'mlp': mlp, 'rf': rf, 'lr': lr, 'lstm': lstm}


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


def rank_heatmap(ranks, model='mlp', timepoint=10):
    f, ax = plt.subplots(figsize=(10, 8))
    feat = model+'_feat'
    data = ranks[ranks['timepoint'] <= timepoint]
    heat = data[['case','timepoint',feat]].\
               groupby(['timepoint',feat])['case'].\
               count().reset_index(name="count")
    heat = heat.pivot(index=feat, columns='timepoint', values='count')
    ax = sns.heatmap(heat.fillna(0), cmap="YlGnBu", annot=True, fmt='.0f',)
    ax.set_title('Model: {}'.format(model.upper()))
    ax.invert_xaxis()
    return


def consensus_heatmap(consensus, timepoint=10):
    f, ax = plt.subplots(figsize=(10, 8))
    data = consensus[consensus['timepoint'] <= timepoint]
    heat = data[['case','timepoint','winner']].\
               groupby(['timepoint','winner'])['case'].\
               count().reset_index(name="count")
    heat = heat.pivot(index='winner', columns='timepoint', values='count')
    ax = sns.heatmap(heat.fillna(0), cmap="YlGnBu", annot=True, fmt='.0f',)
    ax.set_title('Consensus')
    ax.invert_xaxis()
    return
