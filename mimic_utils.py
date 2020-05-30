import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import magec_utils as mg
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from adjustText import adjust_text
from mimic_queries import meds_query, notes_query
from matplotlib.font_manager import FontProperties
from sklearn import svm


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
    try:
        conn = engine.connect()
        conn.execute('SET search_path to ' + schema_name)
        df = pd.read_sql("SELECT * FROM mimic_users_study;", conn)
        conn.close()
        df = df.sort_values(['subject_id', 'timepoint'], ascending=(1, 0))
    except Exception as e:
        raise e
    return df


def get_cohort_meds():
    sqluser = 'postgres'
    dbname = 'mimic'
    schema_name = 'mimiciii'
    engine = create_engine("postgresql+psycopg2://{}:{}@/{}".format(sqluser, sqluser, dbname))
    try:
        conn = engine.connect()
        conn.execute('SET search_path to ' + schema_name)
        df_meds = pd.read_sql(meds_query(prior_hours=48), conn)
        conn.close()
    except Exception as e:
        raise e
    return df_meds


def get_cohort_notes():
    sqluser = 'postgres'
    dbname = 'mimic'
    schema_name = 'mimiciii'
    engine = create_engine("postgresql+psycopg2://{}:{}@/{}".format(sqluser, sqluser, dbname))
    try:
        conn = engine.connect()
        conn.execute('SET search_path to ' + schema_name)
        df_notes = pd.read_sql(notes_query(prior_hours=48), conn)
        conn.close()
    except Exception as e:
        raise e
    return df_notes


def get_pt_admission(case):
    sqluser = 'postgres'
    dbname = 'mimic'
    schema_name = 'mimiciii'
    engine = create_engine("postgresql+psycopg2://{}:{}@/{}".format(sqluser, sqluser, dbname))
    try:
        conn = engine.connect()
        conn.execute('SET search_path to ' + schema_name)
        df_pt = pd.read_sql("select * from admissions where subject_id={};".format(case), conn)
        conn.close()
    except Exception as e:
        raise e
    return df_pt


def get_ml_data(df, outcome='ventilated'):
    df_ml = df.set_index(['subject_id', 'timepoint']).groupby(level=0, group_keys=False). \
        apply(lambda x: featurize(x, outcome)).reset_index()
    return df_ml


def get_ml_series_data(df, outcome='ventilated'):
    df_time = df.set_index(['subject_id']).groupby(level=0, group_keys=False). \
        apply(lambda x: featurize_time(x, outcome)).apply(pd.Series.explode).reset_index()
    return df_time


def train_valid_ml(df_ml, test_size=0.2, seed=7, include_valid=None):
    np.random.seed(seed)

    x_cols = list(set(df_ml.columns) - {'subject_id', 'label'})
    y_cols = ['subject_id', 'label']

    cases = df_ml['subject_id'].unique()

    np.random.shuffle(cases)  # inplace shuffle

    valid_cases = cases[:int(len(cases) * test_size)]
    train_cases = cases[int(len(cases) * test_size):]

    if include_valid is not None:
        valid_set = set(np.load(include_valid))
        valid_cases = list(set(valid_cases).union(valid_set))
        train_cases = list(set(train_cases) - valid_set)

    train_cases = np.isin(df_ml['subject_id'], train_cases)
    valid_cases = np.isin(df_ml['subject_id'], valid_cases)

    xy_train = df_ml.loc[train_cases, :]
    x_train = xy_train[x_cols].copy()
    Y_train = xy_train[y_cols].copy()

    xy_valid = df_ml.loc[valid_cases, :]
    x_valid = xy_valid[x_cols].copy()
    Y_valid = xy_valid[y_cols].copy()

    x_train = impute(x_train)
    x_valid = impute(x_valid)

    stsc = StandardScaler()
    xst_train = stsc.fit_transform(x_train)
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=x_train.columns)

    xst_valid = stsc.transform(x_valid)
    xst_valid = pd.DataFrame(xst_valid, index=x_valid.index, columns=x_valid.columns)

    return x_train, x_valid, stsc, xst_train, xst_valid, Y_train, Y_valid


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
    series_means = df_series_train_X.mean()
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

    return stsc, series_means, df_series_train, df_series_valid, xt_train, Yt_train, xt_valid, Yt_valid


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
    print('Built MLP classifier!')

    sv = svm.SVC(probability=True, class_weight="balanced")
    sv.fit(xst_train, Y_train['label'])
    print('Built SVM classifier!')

    # rf = CalibratedClassifierCV(RandomForestClassifier(n_estimators=800,
    #                                                    min_samples_split=2,
    #                                                    min_samples_leaf=4,
    #                                                    max_features='sqrt',
    #                                                    max_depth=90,
    #                                                    bootstrap=True,
    #                                                    n_jobs=-1,
    #                                                    class_weight="balanced"),
    #                             method='sigmoid', cv=5)
    # rf.fit(xst_train, Y_train['label'])

    lr = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs')
    lr.fit(xst_train, Y_train['label'])
    print('Built LR classifier!')

    lstm = KerasClassifier(build_fn=create_lstm, epochs=100, batch_size=64, verbose=0)
    lstm.fit(xt_train, Yt_train, epochs=100, batch_size=64, verbose=0)
    print('Built LSTM classifier!')

    return {'mlp': mlp, 'svm': sv, 'lr': lr, 'lstm': lstm}


def mimic_ensemble_metrics(models, xst_validation, Y_validation, xt_valid, df_series_valid, verbose=False):
    assert 'lstm' in models, 'missing lstm model'

    others = list()
    for k in models.keys():
        if k == 'lstm':
            lstm = models[k]
            lstm_preds = pd.DataFrame(lstm.predict_proba(xt_valid)[:, 1])
            lstm_preds.columns = ['lstm_prob_1']
            lstm_labels = pd.DataFrame(df_series_valid.groupby(level='case', group_keys=False)['label']. \
                                       agg('first')).reset_index()
            lstm_preds = pd.concat([lstm_preds, lstm_labels], axis=1)
        else:
            model = models[k]
            preds = pd.DataFrame(model.predict_proba(xst_validation)[:, 1])
            preds.columns = [k + '_prob_1']
            others.append(preds)

    assert len(others), "non-lstm models not in models dictionaty..."

    preds = pd.concat(others + [Y_validation.reset_index().drop('index', 1)], axis=1)
    preds = preds.rename(columns={"subject_id": "case"})
    preds['label'] = preds['label'].astype(int)

    assert set(preds.case.unique()) == set(lstm_preds.case.unique()), "cannot join LSTM and Other models"

    preds = preds.set_index('case')
    lstm_preds = lstm_preds.set_index('case')
    preds = preds.merge(lstm_preds['lstm_prob_1'], left_index=True, right_index=True)

    preds['ensemble_prob'] = preds[[k + '_prob_1' for k in models.keys()]].mean(axis=1)

    preds['class_1'] = (preds['ensemble_prob'] > 0.5).astype(int)

    accuracy, precision, recall, f1, auc = mg.model_metrics(preds['ensemble_prob'],
                                                            preds['class_1'],
                                                            preds['label'],
                                                            verbose=verbose)

    return accuracy, precision, recall, f1, auc


def last_val(x):
    vals = x[~np.isnan(x)]
    if len(vals):
        return vals[-1]
    else:
        return None


def featurize_time(df, outcome):
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
        out['label'] = [int(x) for x in df[outcome].values]
    return pd.Series(out)


def featurize(df, outcome):
    out = dict()
    for lab in labs:
        out[lab] = last_val(df[lab])
    for vital in vitals:
        out[vital] = last_val(df[vital])
    for comob in comobs:
        out[comob] = last_val(df[comob])
    for other in others:
        out[other] = last_val(df[other])
    out['label'] = int(df[outcome].iloc[-1])
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


def plot_risk(ax, x, y, z, w, yy, case, label):
    ax.plot(x, y, 'rx--')
    ax.plot(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10), 0.5 * np.ones(10), '--')
    txt = 'Case {}: Hourly Estimated Ensemble Risk (Outcome: {})'
    ax.set_title(txt.format(case, label))
    ax.set_ylabel('Ensemble Risk')
    ax.set_xlabel('hours to event')
    ax.grid('on')
    # ax.set_ylim([0.2, 0.9])

    texts = []

    for i, txt in enumerate(z):
        if np.isnan(yy[i]):
            continue
        if (w[i] < 0.5 < y[i]):
            msg = txt + ' = {:.2f}\n perturb. risk = {:.2g}'.format(yy[i], w[i])
            texts.append(ax.text(x[i], y[i], msg))

    f = interpolate.interp1d(x, y)
    x = np.linspace(min(x), max(x), 140)
    y = f(x)
    adjust_text(texts, x, y, arrowprops=dict(arrowstyle="->", color='b', lw=0.5),
                autoalign='xy')
    return


def plot_feature(ax, x, y, best_feat, case, label, title=''):
    ax.plot(x, y, 'bo--')
    txt = 'Case {}: Hourly average {}  (Outcome={})'
    ax.set_title(txt.format(case, best_feat if not title else title, label))
    ax.set_ylabel(best_feat)
    ax.set_xlabel('hours to event')
    ax.invert_xaxis()
    ax.tick_params(labelbottom=True)
    ax.grid('on')
    return


def best_feat_plot(joined, cohort, index, title='', save=False):
    data = joined.loc[index]
    case, t_0, label, orig_prob, new_risk = data[['case', 'timepoint', 'label',
                                                  'orig_prob_ensemble', 'new_risk']]
    best_feat = joined.loc[index]['best_feat']

    xy = cohort[cohort['subject_id'] == case][['timepoint', best_feat]].values
    x = [int(x[0]) for x in xy]
    yy = [x[1] for x in xy]

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 12))

    plot_feature(ax[0], x, yy, best_feat, case, label, title=title)

    xyzw = joined[joined.case == case][['timepoint', 'orig_prob_ensemble', 'best_feat', 'new_risk']].values
    x = [int(x[0]) for x in xyzw]
    y = [x[1] for x in xyzw]
    z = [x[2] for x in xyzw]
    w = [x[3] for x in xyzw]
    zz = [cohort[(cohort['subject_id'] == case) & (cohort['timepoint'] == x[i])][feat].values[0]
          for i, feat in enumerate(z)]

    plot_risk(ax[1], x, y, z, w, zz, case, label)

    if save:
        plt.savefig('case_{}_series.png'.format(case))

    return fig, ax


def best_feature(data, cols, feat=None):
    models = list(set([c.split('_')[-1] for c in cols]))
    if feat is not None:
        features = [feat]
    else:
        features = list(set(['_'.join(c.split('_')[1:-2]) for c in cols]))
    best_feat = ''
    new_risk = None
    for feat in features:
        feat_risks = []
        orig_probs = []
        for model in models:
            magec = model + '_' + feat
            perturb = 'perturb_' + feat + '_prob_' + model
            orig = 'orig_prob_' + model
            if magec in data and perturb in data and orig in data:
                perturb = data[perturb]
                orig = data[orig]
                feat_risks.append(perturb)
                orig_probs.append(orig)
        # model average
        feat_risk = np.mean(feat_risks)
        orig_prob = np.mean(orig_probs)
        if new_risk is None or feat_risk < new_risk and feat_risk < orig_prob:
            new_risk = feat_risk
            best_feat = feat
    return pd.Series((best_feat, new_risk), index=['best_feat', 'new_risk'])


def full_panel_plot(train_cols, features, stsc, joined, cohort, index, label='Outcome',
                    models=('lr', 'rf', 'mlp', 'lstm'),
                    limit=None, rotate=None, save=None, title=None, magec_ensemble=False):
    tmp = joined.loc[index]
    case = tmp.case
    timepoint = tmp.timepoint

    data = joined.loc[(joined.case == case) & (joined.timepoint == timepoint)]
    case_df = mg.case_stats(data, case, timepoint, models=models)

    if limit is not None:
        topK = case_df.groupby('feature')['risk_new'].mean().sort_values()[:limit].index.values
        case_df = case_df[np.isin(case_df['feature'], topK)]
        train_cols_idx = [train_cols.to_list().index(x) for x in topK]
        features = topK
    else:
        train_cols_idx = [i for i in range(len(train_cols))]

    fig = plt.figure(figsize=(16, 12))
    grid = plt.GridSpec(9, 7, wspace=0.2, hspace=0.1)

    series_fig1 = fig.add_subplot(grid[:3, :3])
    series_fig2 = fig.add_subplot(grid[:3, 3:])
    main_fig = fig.add_subplot(grid[4, :3])
    ml_fig = fig.add_subplot(grid[6, :3])
    bar_fig = fig.add_subplot(grid[3:7, 3:])
    mg_fig = fig.add_subplot(grid[8, :])

    base = case_df.groupby('feature')['risk'].mean()
    bar_fig = sns.barplot(x="feature", y="risk_new", data=case_df, ci=None, ax=bar_fig)
    bar_fig.plot(np.linspace(bar_fig.get_xlim()[0], bar_fig.get_xlim()[1], 10),
                 np.mean(base.values) * np.ones(10), '--')
    bar_fig.legend(['current risk', 'estimated risk'], loc='upper right')
    bar_fig.set_ylabel('estimated risk')
    bar_fig.set_ylim([0, min(round(1.2 * bar_fig.get_ylim()[1], 1), 1)])
    if rotate is not None:
        bar_fig.set_xticklabels(bar_fig.get_xticklabels(), rotation=rotate)
    bar_fig.set_xlabel('')

    collabel0 = ["Case", str(case)]

    cell_feat = [feat for feat in train_cols[train_cols_idx]]
    cell_vals = [round(val, 3) for val in stsc.inverse_transform(data[train_cols])[0][train_cols_idx]]
    celldata0 = [[x[0], x[1]] for x in zip(cell_feat, cell_vals)] + [['True Outcome', data[label].values[0]]]

    collabel1 = ["Model", "Predicted Risk"]

    celldata1 = [[model.upper(), round(data['orig_prob_' + model].values[0], 3)] for model in models]

    collabel2 = ["Model"] + ["MAgEC " + feat for feat in features]  # + ["Sum"]

    celldata2 = list()

    if not magec_ensemble:
        models = [m for m in models if m != 'ensemble']

    for model in models:
        add_model = True
        line = list()
        for feat in features:
            f = model + '_' + feat
            if f not in data:
                add_model = False
                break
            else:
                line.append(round(data[f].values[0], 3))
        if add_model:
            celldata2.append([model.upper()] + line)

    celldata2_sum = np.sum(np.array([t[1:] for t in celldata2]), axis=0)
    celldata2.append(['SUM'] + [round(t, 2) for t in celldata2_sum])

    main_fig.axis('tight')
    main_fig.axis('off')
    ml_fig.axis('tight')
    ml_fig.axis('off')
    mg_fig.axis('tight')
    mg_fig.axis('off')

    table0 = main_fig.table(cellText=celldata0, colLabels=collabel0, loc='center', cellLoc='center')
    table1 = ml_fig.table(cellText=celldata1, colLabels=collabel1, loc='center', cellLoc='center')
    table2 = mg_fig.table(cellText=celldata2, colLabels=collabel2, loc='center', cellLoc='center')

    table0.set_fontsize(12)
    table0.scale(1.5, 1.5)

    table1.set_fontsize(12)
    table1.scale(1.5, 1.5)

    table2.set_fontsize(12)
    table2.scale(1.5, 1.5)

    table0.auto_set_column_width(col=list(range(len(collabel0))))
    table1.auto_set_column_width(col=list(range(len(collabel1))))
    table2.auto_set_column_width(col=list(range(len(collabel2))))

    mg.bold_column(table0)
    mg.bold_column(table1)
    mg.bold_column(table2)

    for (row, col), cell in table2.get_celld().items():
        if row == len(models) + 1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    if save is not None:
        plt.savefig(str(save) + '.png', bbox_inches='tight')

    return fig


def print_notes(df_notes, case):
    print('\n***NEXT NOTE***\n\n'.join(df_notes[df_notes.subject_id == case].\
                                       sort_values('charttime')['text'].values.tolist()))
