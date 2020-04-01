import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from numpy import interp
from collections import OrderedDict


def get_logit(prob, eps=1e-16):
    return np.log2((prob+eps)/(1-prob+eps))


def get_column_categories(table, sort=False):
    cats = OrderedDict()
    for kk,vv in table.dtypes.items():
        if vv is np.dtype('O'):
            cats[kk] = table[kk].fillna('').unique().tolist()
            if sort:
                cats[kk] = sorted(cats[kk])
        else:
            cats[kk] = []
    return cats


def hier_col_name_generator(categories):
    for cl, vv in categories.items():
        if len(vv) > 0:
            for cat in vv:
                yield '{}-{}'.format(cl, cat) if len(vv) > 0 else vv
        else:
            yield cl


def predict(model, data):
    """
    Model output (predicted) probabilities.
    Wrapper for predict_proba function in scikit-learn models.
    When a model does not have a predict_proba use predict interface.
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data)
        if probs.shape[1] == 2:
            probs = probs[:, 1].ravel()
        else:
            probs = probs.ravel()
    else:
        probs = np.array(model.predict(data))
    return probs


def z_perturbation(model, target_data,
                   score_preprocessing=get_logit,
                   score_comparison=lambda x_baseline, x: x - x_baseline,
                   sort_categories=True,
                   categories=None,
                   features=None,
                   binary=None,
                   timepoint_level='timepoint',
                   epsilon_value=0):
    '''
    Main method for computing a MAgEC. Assumes 'scaled/normalized' features in target data.

    Supporting 2 types of variables:
    - numeric / floats
    - binary / boolean
    Default score_comparison subtracts perturbed output from original.
    For a binary classification task, where 1 denotes a "bad" outcome, a good perturbation
    is expected to result in a negative score_comparison (assuming monotonic score_preprocessing).
    '''
    probs_orig = predict(model, target_data)
    logit_orig = score_preprocessing(probs_orig)
    logit_data = target_data.copy()
    logit_data_cols = logit_data.columns
    logit_data['logit_orig'] = logit_orig
    logit_data['probs_orig'] = probs_orig
    logit_data = logit_data.drop(logit_data_cols, axis=1)

    if features is None:
        features = target_data.columns.unique()
    else:
        features = np.asarray(features)

    if binary is None:
        binary = target_data.apply(lambda x: len(np.unique(x)), ) <= 2
        binary = binary[binary].index.tolist()

    if categories is None:
        categories = get_column_categories(target_data[features], sort=sort_categories)

    timepoints = target_data.index.get_level_values(timepoint_level).unique()
    cases = target_data.index.to_frame().drop('timepoint', axis=1).drop_duplicates().values

    n_case_inds = cases.shape[1]

    prob_deltas_per_cell = pd.DataFrame(index=target_data.index,
                                        columns=pd.Index(hier_col_name_generator(categories),
                                                         name='features'))
    for tt in timepoints:
        for var_name in target_data.columns:

            idx = tuple([(slice(None))] * n_case_inds + [tt])

            if var_name in binary:
                epsilon = target_data[var_name].value_counts().idxmax()  # most frequent value
            else:
                epsilon = epsilon_value

            target_data_pert = target_data.copy()

            target_data_pert.loc[idx, var_name] = epsilon  # z-value (=0 for numerical values)

            probs = predict(model, target_data_pert.loc[idx, :])

            logit = score_preprocessing(probs)

            logit_orig_sliced = logit_data.loc[idx, 'logit_orig']
            logit_diff = score_comparison(logit_orig_sliced, logit)

            prob_deltas_per_cell.loc[idx, var_name] = logit_diff

            prob_deltas_per_cell.loc[idx, 'orig_prob'] = logit_data.loc[idx, 'probs_orig']
            prob_deltas_per_cell.loc[idx, 'perturb_{}_prob'.format(var_name)] = probs

    return prob_deltas_per_cell.astype(float)


def m_prefix(magecs, feature, model_name=None):
    """
    Given a feature (e.g. BMI) and a magecs dataframe extract prefix (model_name).
    """
    prefix = 'm'
    for c in magecs.columns:
        splits = c.split('_')
        if len(splits) > 1 and feature == '_'.join(splits[1:]):
            prefix = splits[0]
            if model_name is not None:
                assert prefix == model_name
            break
    return prefix


def create_magec_col(model_name, feature):
    return model_name + '_' + feature


def case_magecs(model, data, epsilon_value=0, model_name=None):
    """
    Compute MAgECs for every 'case' (individual row/member table).
    Use all features in data to compute MAgECs.
    NOTE 1: we prefix MAgECs with model_name.
    NOTE 2: we postfix non-MAgECs, such as 'perturb_<FEAT>_prob' with model_name.
    """
    magecs = z_perturbation(model, data, epsilon_value=epsilon_value)
    features = magecs.columns
    magecs = magecs.reset_index()
    # rename features in case_magecs to reflect the fact that they are derived for a specific model
    prefix = 'm' if model_name is None else model_name
    postfix = prefix
    for feat in features:
        if feat == 'orig_prob' or (feat[:8] == 'perturb_' and feat[-5:] == '_prob'):
            magecs.rename(columns={feat: feat + '_' + postfix}, inplace=True)
        else:
            magecs.rename(columns={feat: create_magec_col(prefix, feat)}, inplace=True)
    return magecs


def normalize_magecs(magecs,
                     features=None,
                     model_name=None):
    """
    Normalize MAgECs for every 'case' using an L2 norm.
    Use (typically) all MAgEC columns (or a subset of features). Former is advised.
    NOTE: The convention is that MAgECs are prefixed with model_name.
    """
    out = magecs.copy()

    if features is None:
        prefix = 'm_' if model_name is None else model_name + '_'
        cols = [c for c in magecs.columns if c.startswith(prefix)]
    else:
        cols = [create_magec_col(m_prefix(magecs, feat, model_name), feat) for feat in features]

    for (idx, row) in out.iterrows():
        norm = np.linalg.norm(row.loc[cols].values)
        out.loc[idx, cols] = out.loc[idx, cols] / norm
    return out


def plot_all_violin(magecs,
                    features=('Age', 'BloodPressure', 'BMI', 'Glucose', 'Insulin',
                              'SkinThickness', 'DiabetesPedigreeFunction'),
                    model_name=None):
    """
    Violin plots for MAgECs.
    """
    colors = list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots(nrows=1, ncols=len(features), figsize=(16, 10))
    ymin = 10
    ymax = -10
    for i, feat in enumerate(features):
        mfeat = m_prefix(magecs, feat, model_name) + "_" + feat
        ymin = min(np.min(magecs[mfeat]), ymin)
        ymax = max(np.max(magecs[mfeat]), ymax)
        sns.violinplot(y=magecs[mfeat], ax=ax[i], color=colors[i % len(colors)])
        ax[i].grid('on')
        ax[i].set_ylabel('')
        ax[i].set_title(feat)
    for i in range(len(features)):
        ax[i].set_ylim([1.5 * ymin, 1.5 * ymax])


def magec_cols(magec, features):
    all_cols = magec.columns
    orig_prob_col = [col for col in all_cols if col.startswith('orig_prob_')]
    jcols = ['case', 'timepoint']
    m_cols = [col for col in all_cols if '_'.join(col.split('_')[1:]) in features]
    prob_cols = [col for col in all_cols if col.startswith('perturb_') and
                 col[8:].split('_prob_')[0] in features]
    cols = jcols + m_cols + prob_cols + orig_prob_col
    return jcols, cols


def magec_models(*magecs,
                 Xdata=None,
                 Ydata=None,
                 features=('Age', 'BloodPressure', 'BMI', 'Glucose', 'Insulin',
                           'SkinThickness', 'DiabetesPedigreeFunction')):
    """
    Wrapper function for joining MAgECs from different models together and (optionally) w/ tabular data
    """
    assert len(magecs) > 1
    jcols, cols = magec_cols(magecs[0], features)
    magec = magecs[0][cols]
    if Xdata is not None:
        magec = magec.merge(Xdata.reset_index(), left_on=jcols, right_on=jcols)
    if Ydata is not None:
        magec = magec.merge(Ydata.reset_index(), left_on=jcols, right_on=jcols)
    for mgc in magecs[1:]:
        _, cols = magec_cols(mgc, features)
        mgc = mgc[cols]
        magec = magec.merge(mgc, left_on=jcols, right_on=jcols)
    return magec


def magec_rank(magecs,
               models=('mlp', 'rf', 'lr'),
               rank=3,
               features=('BloodPressure', 'BMI', 'Glucose', 'Insulin', 'SkinThickness'),
               outcome='Outcome'):
    """
    Compute top-magecs (ranked) for each model for each 'case/timepoint' (individual row in tabular data).
    Input is a list of one or more conputed magecs given a model.
    Output is a Pandas dataframe with computed magecs, filtering out positive magecs.
    Positive magecs indicate counter-productive interventions.
    """
    ranks = {}

    # each row contains all MAgEC coefficients for a 'case/timepoint'
    for (idx, row) in magecs.iterrows():
        model_ranks = {}
        if outcome in row:
            key = (row['case'], row['timepoint'], row[outcome])
        else:
            key = (row['case'], row['timepoint'])
        for model in models:
            # initialize all models coefficients (empty list)
            model_ranks[model] = list()
        for col in features:
            # iterate of all features
            for model in models:
                # each model should contain a corresponding magec
                feat = create_magec_col(model, col)
                assert feat in row, "feature {} not in magecs".format(feat)
                magec = row[feat]
                # we are using a priority queue for the magec coefficients
                # heapq is a min-pq, we are reversing the sign so that we can use a max-pq
                if len(model_ranks[model]) < rank:
                    heapq.heappush(model_ranks[model], (-magec, col))
                else:
                    _ = heapq.heappushpop(model_ranks[model], (-magec, col))
                    # store magecs (top-N where N=rank) for each key ('case/timepoint')
        ranks[key] = model_ranks
        # create a Pandas dataframe with all magecs for a 'case/timepoint'
    out = list()
    out_col = None
    columns = []
    for k, v in ranks.items():
        if len(k) == 3:
            l = [k[0], k[1], k[2]]
            if out_col is None:
                out_col = outcome
                columns = ['case', 'timepoint', outcome]
        else:
            l = [k[0], k[1]]
            if not len(columns):
                columns = ['case', 'timepoint']
        for model in models:
            while v[model]:  # retrieve priority queue's magecs (max-pq with negated (positive) magecs)
                magec, feat = heapq.heappop(v[model])
                if magec < 0:  # negative magecs are originally positive magecs and are filtered out
                    l.append(None)
                    l.append("not_found")
                else:
                    l.append(-magec)  # retrieve original magec sign
                    l.append(feat)
        out.append(l)

    out = pd.DataFrame.from_records(out)
    # create dataframe's columns
    for model in models:
        if rank == 1:
            columns.append(model + '_magec')
            columns.append(model + '_feat')
        else:
            for r in range(rank, 0, -1):
                columns.append(model + '_magec_{}'.format(r))
                columns.append(model + '_feat_{}'.format(r))
    out.columns = columns
    out['case'] = out['case'].astype(magecs['case'].dtype)
    out['timepoint'] = out['timepoint'].astype(magecs['timepoint'].dtype)
    if out_col:
        out[out_col] = out[out_col].astype(magecs[out_col].dtype)

    pert_cols = ['perturb_' + col + '_prob' + '_' + model for col in features for model in models]
    orig_cols = ['orig_prob_' + model for model in models]
    all_cols = ['case', 'timepoint'] + pert_cols + orig_cols + features
    out = out.merge(magecs[all_cols],
                    left_on=['case', 'timepoint'],
                    right_on=['case', 'timepoint'])
    return out


def print_ranks_stats(ranks, models=('mlp', 'rf', 'lr')):
    columns = ranks.columns
    for model in models:
        cols = [col for col in columns if col.startswith(model + '_' + 'feat')]
        if len(cols):
            print("\t {} MAgEC Stats".format(model))
            for col in cols:
                print("**** " + col + " ****")
                print(ranks[col].value_counts())
                print("***********")


def magec_consensus(magec_ranks,
                    models=('mlp', 'rf', 'lr'),
                    use_weights=False,
                    weights={'rf': None, 'mlp': None, 'lr': None},
                    outcome='Outcome',
                    policy='sum'):
    """
    Given a ranked list of magecs from one or more models compute a single most-important magec.
    There are 2 types of "MAgEC" columns in magec_ranks:
    1. 'feat_' with the name of the magec feature
    2. 'magec_' with the value of the magec
    The prefix in the column name indicates the model used (e.g. 'mlp_').
    The column names end in _{rank_number} when rank > 1.
    INPUT magec_ranks EXAMPLE when rank > 1
        case                                          2
        timepoint                                     0
        Outcome                                       1
        mlp_magec_3                                 NaN
        mlp_feat_3                            not_found
        mlp_magec_2                                 NaN
        mlp_feat_2                            not_found
        mlp_magec_1                            -0.16453
        mlp_feat_1                        SkinThickness
        rf_magec_3                           -0.0511598
        rf_feat_3                         SkinThickness
        rf_magec_2                            -0.152834
        rf_feat_2                         BloodPressure
        rf_magec_1                            -0.282895
        rf_feat_1                               Glucose
        lr_magec_3                           -0.0158208
        lr_feat_3                         SkinThickness
        lr_magec_2                           -0.0356751
        lr_feat_2                               Insulin
        lr_magec_1                            -0.614731
        lr_feat_1                               Glucose
        perturb_BloodPressure_prob_mlp         0.658987
        perturb_BloodPressure_prob_rf          0.433044
        perturb_BloodPressure_prob_lr          0.784033
        perturb_BMI_prob_mlp                   0.745703
        perturb_BMI_prob_rf                    0.802258
        perturb_BMI_prob_lr                    0.822721
        perturb_Glucose_prob_mlp               0.604908
        perturb_Glucose_prob_rf                0.383129
        perturb_Glucose_prob_lr                0.493356
        perturb_Insulin_prob_mlp               0.618258
        perturb_Insulin_prob_rf                0.484568
        perturb_Insulin_prob_lr                0.721596
        perturb_SkinThickness_prob_mlp         0.517638
        perturb_SkinThickness_prob_rf          0.473091
        perturb_SkinThickness_prob_lr          0.728289
        orig_prob_mlp                          0.591666
        orig_prob_rf                           0.493407
        orig_prob_lr                           0.733549
        BloodPressure                           1.11285
        BMI                                   -0.697264
        Glucose                                0.992034
        Insulin                               -0.435478
        SkinThickness                          0.779127

    INPUT magec_ranks EXAMPLE when rank = 1
        case                                          2
        timepoint                                     0
        Outcome                                       1
        mlp_magec                              -0.16453
        mlp_feat                          SkinThickness
        rf_magec                              -0.282895
        rf_feat                                 Glucose
        lr_magec                              -0.614731
        lr_feat                                 Glucose
        perturb_BloodPressure_prob_mlp         0.658987
        perturb_BloodPressure_prob_rf          0.433044
        perturb_BloodPressure_prob_lr          0.784033
        perturb_BMI_prob_mlp                   0.745703
        perturb_BMI_prob_rf                    0.802258
        perturb_BMI_prob_lr                    0.822721
        perturb_Glucose_prob_mlp               0.604908
        perturb_Glucose_prob_rf                0.383129
        perturb_Glucose_prob_lr                0.493356
        perturb_Insulin_prob_mlp               0.618258
        perturb_Insulin_prob_rf                0.484568
        perturb_Insulin_prob_lr                0.721596
        perturb_SkinThickness_prob_mlp         0.517638
        perturb_SkinThickness_prob_rf          0.473091
        perturb_SkinThickness_prob_lr          0.728289
        orig_prob_mlp                          0.591666
        orig_prob_rf                           0.493407
        orig_prob_lr                           0.733549
        BloodPressure                           1.11285
        BMI                                   -0.697264
        Glucose                                0.992034
        Insulin                               -0.435478
        SkinThickness                          0.779127
    """

    cols = list(set(magec_ranks.columns) - {'case', 'timepoint', outcome})

    def name_matching(cols, models):
        # get all magec column names
        col_names = dict()
        for col in cols:
            prefix = col.split('_')[0]
            if prefix in models:
                if prefix in col_names:
                    col_names[prefix].append(col)
                else:
                    col_names[prefix] = [col]
        # magec/feat column names come in pairs
        magecs_feats = dict()
        for model, cols in col_names.items():
            feat2magic = dict()
            assert len(cols) % 2 == 0, "magec/feat cols should come in pairs"
            if len(cols) == 2:
                if 'feat' in cols[0] and 'magec' in cols[1]:
                    feat2magic[cols[0]] = cols[1]
                elif 'feat' in cols[1] and 'magec' in cols[0]:
                    feat2magic[cols[1]] = cols[0]
                else:
                    raise ValueError('magec/feat substring not present in column names')
            else:
                # reversed names sorted (e.g. 1_taef_plm)
                feats = sorted([col[::-1] for col in cols if 'feat' in col])
                # reversed names sorted (e.g. 1_cegam_plm)
                magecs = sorted([col[::-1] for col in cols if 'magec' in col])
                assert len(feats) == len(cols) / 2, "'feat' substring missing in column name"
                assert len(magecs) == len(cols) / 2, "'magec' substring missing in column name"
                for i, feat in enumerate(feats):
                    feat2magic[feat[::-1]] = magecs[i][::-1]
            # return dictionary with magec feature column names and magec value column name for every model
            magecs_feats[model] = feat2magic
        return magecs_feats

    magecs_feats = name_matching(cols, models)

    out = list()
    for (idx, row) in magec_ranks.iterrows():
        member = list()
        key = row['case'], row['timepoint']
        winner = magec_winner(magecs_feats, row, use_weights=use_weights, weights=weights, policy=policy)
        member.append(key[0])
        member.append(key[1])
        winner_feat = None if winner is None else winner[0]
        winner_score = None if winner is None else winner[1]
        winner_consensus = None if winner is None else winner[2]
        winner_models = None if winner is None else winner[3]
        winner_prob_ratio_all = None
        winner_prob_ratio = None
        if winner_feat is not None:
            all_model_ratios = list()
            model_ratios = list()
            for model in magecs_feats.keys():
                orig_prob = row['orig_prob_' + model]
                feat_prob = row['perturb_' + winner_feat + '_prob_' + model]
                ratio = 100*(orig_prob-feat_prob) / orig_prob
                all_model_ratios.append(ratio)
                if model in winner_models:
                    model_ratios.append(ratio)
            winner_prob_ratio_all = np.mean(all_model_ratios)
            winner_prob_ratio = np.mean(model_ratios)
        member.append(winner_feat)
        member.append(winner_score)
        member.append(winner_consensus)
        member.append(winner_models)
        member.append(winner_prob_ratio)
        member.append(winner_prob_ratio_all)
        out.append(member)
    out = pd.DataFrame.from_records(out)
    out.columns = ['case', 'timepoint',
                   'winner', 'score',
                   'consensus', 'models',
                   'avg_percent_consensus', 'avg_percent_all']
    return out


def magec_winner(magecs_feats,
                 row,
                 scoring=lambda w: abs(w),
                 use_weights=False,
                 weights={'rf': None, 'mlp': None, 'lr': None},
                 policy='sum'):
    """
    Compute MAgEC winner from a list of MAgECs from one or more models for a single 'case/timepoint'.
    magecs_feats is a dictionary with magec feature column names and magec value column names for every model,
     e.g
    {'rf': {'rf_feat_1': 'rf_magec_1', 'rf_feat_2': 'rf_magec_2'},
     'mlp': {'mlp_feat_1': 'mlp_magec_1', 'mlp_feat_2': 'mlp_magec_2'},
     'lr': {'lr_feat_1': 'lr_magec_1', 'lr_feat_2': 'lr_magec_2'}}
    """

    assert policy in ['sum', 'mean'], "Only 'sum' or 'mean' policy is supported"

    winner = None
    consensus = {}
    scores = {}
    if use_weights:
        assert sorted(weights.keys()) == sorted(magecs_feats.keys())
    for model, feat_dict in magecs_feats.items():
        for feat_col, score_col in feat_dict.items():
            feat = row[feat_col]
            score = row[score_col]
            if not np.isnan(score):
                score = scoring(score)
                if use_weights:
                    if weights[model] is not None:
                        score *= weights[model]
                if feat in scores:
                    scores[feat] += score
                    consensus[feat].add(model)
                else:
                    scores[feat] = score
                    consensus[feat] = {model}
    # get consensus
    for feat, score in scores.items():
        if policy == 'mean':
            score /= len(consensus[feat])
        if winner is None or score > winner[1]:
            winner = (feat, score, len(consensus[feat]), sorted(list(consensus[feat])))

    return winner


def magec_similarity(case_magecs,
                     x_validation_p,
                     features=('BloodPressure', 'BMI', 'Glucose', 'Insulin',
                               'SkinThickness', 'DiabetesPedigreeFunction'),
                     model_name=None):
    if model_name is None:
        model_name = "m"
    cols = [model_name + "_" + feat for feat in features] + list(features) + ['Outcome']
    df = case_magecs.merge(x_validation_p, left_on=['case', 'timepoint'], right_index=True)[cols]
    return df


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def pdf(data):
    # Empirical average and variance
    avg = np.mean(data)
    var = np.var(data)
    # Gaussian PDF
    pdf_x = np.linspace(np.min(data), np.max(data), 100)
    pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)
    return pdf_x, pdf_y, avg, var


def plot_train_valid(train, valid, feature):
    train = train[feature]
    valid = valid[feature]
    pdf_x_t, pdf_y_t, avg_t, var_t = pdf(train)
    pdf_x_v, pdf_y_v, avg_v, var_v = pdf(valid)
    # Figure
    plt.figure(figsize=(10, 8))
    plt.hist(train, 30, density=True, alpha=0.5)
    plt.hist(valid, 30, density=True, alpha=0.5)
    plt.plot(pdf_x_t, pdf_y_t, 'b--')
    plt.plot(pdf_x_v, pdf_y_v, 'g--')
    plt.legend(["train fit", "valid fit", "train", "valid"])
    plt.title("mean ({:.2g}, {:.2g}), std ({:.2g}, {:.2g})".format(avg_t, avg_v, np.sqrt(var_t), np.sqrt(var_v)))
    plt.show()


def plot_feature(df, feature, ax=None):
    data = df[feature]
    pdf_x, pdf_y, avg, var = pdf(data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(data, 30, density=True, alpha=0.5)
    ax.plot(pdf_x, pdf_y, 'b--')
    ax.set_title("{} (mean: {:.2g}, std: {:.2g})".format(feature, avg, np.sqrt(var)))
    return ax


def plot_pima_features(df):
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax[0, 0] = plot_feature(df, 'BMI', ax[0, 0])
    ax[0, 1] = plot_feature(df, 'BloodPressure', ax[0, 1])
    ax[1, 0] = plot_feature(df, 'DiabetesPedigreeFunction', ax[1, 0])
    ax[1, 1] = plot_feature(df, 'Glucose', ax[1, 1])
    ax[2, 0] = plot_feature(df, 'Insulin', ax[2, 0])
    ax[2, 1] = plot_feature(df, 'SkinThickness', ax[2, 1])
    return ax


def model_performance(model, X, y, subtitle):
    # Kfold
    cv = KFold(n_splits=5, shuffle=False, random_state=42)
    y_real = []
    y_proba = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1

    for train, test in cv.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        pred_proba = model.predict_proba(X.iloc[test])
        precision, recall, _ = precision_recall_curve(y.iloc[test], pred_proba[:, 1])
        y_real.append(y.iloc[test])
        y_proba.append(pred_proba[:, 1])
        fpr, tpr, t = roc_curve(y[test], pred_proba[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_matrix = confusion_matrix(y, y_pred)
    trace1 = go.Heatmap(z=conf_matrix, x=["0 (pred)", "1 (pred)"],
                        y=["0 (true)", "1 (true)"], xgap=2, ygap=2,
                        colorscale='Viridis', showscale=False)

    # Show metrics
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    tn = conf_matrix[0, 0]
    Accuracy = ((tp + tn) / (tp + tn + fp + fn))
    Precision = (tp / (tp + fp))
    Recall = (tp / (tp + fn))
    F1_score = (2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=(show_metrics[0].values),
                    y=['Accuracy', 'Precision', 'Recall', 'F1_score'], text=np.round_(show_metrics[0].values, 4),
                    textposition='auto', textfont=dict(color='black'),
                    orientation='h', opacity=1, marker=dict(
            color=colors,
            line=dict(color='#000000', width=1.5)))

    # Roc curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    trace3 = go.Scatter(x=mean_fpr, y=mean_tpr,
                        name="Roc : ",
                        line=dict(color=('rgb(22, 96, 167)'), width=2), fill='tozeroy')
    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('black'), width=1.5,
                                  dash='dot'))

    # Precision - recall curve
    y_real = y
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    trace5 = go.Scatter(x=recall, y=precision,
                        name="Precision" + str(precision),
                        line=dict(color=('lightcoral'), width=2), fill='tozeroy')

    mean_auc = round(mean_auc, 3)

    # Subplots
    fig = tls.make_subplots(rows=2, cols=2, print_grid=False,
                            specs=[[{}, {}],
                                   [{}, {}]],
                            subplot_titles=('Confusion Matrix',
                                            'Metrics',
                                            'ROC curve' + " " + '(' + str(mean_auc) + ')',
                                            'Precision - Recall curve',
                                            ))
    # Trace and layout
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)

    fig['layout'].update(showlegend=False, title='<b>Model performance report (5 folds)</b><br>' + subtitle,
                         autosize=False, height=830, width=830,
                         plot_bgcolor='black',
                         paper_bgcolor='black',
                         margin=dict(b=195), font=dict(color='white'))
    fig["layout"]["xaxis1"].update(color='white')
    fig["layout"]["yaxis1"].update(color='white')
    fig["layout"]["xaxis2"].update((dict(range=[0, 1], color='white')))
    fig["layout"]["yaxis2"].update(color='white')
    fig["layout"]["xaxis3"].update(dict(title="false positive rate"), color='white')
    fig["layout"]["yaxis3"].update(dict(title="true positive rate"), color='white')
    fig["layout"]["xaxis4"].update(dict(title="recall"), range=[0, 1.05], color='white')
    fig["layout"]["yaxis4"].update(dict(title="precision"), range=[0, 1.05], color='white')
    for i in fig['layout']['annotations']:
        i['font'] = titlefont = dict(color='white', size=14)
    py.iplot(fig)


def scores_table(model, X, y, subtitle):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    res = []
    for sc in scores:
        scores = cross_val_score(model, X, y, cv=5, scoring=sc)
        res.append(scores)
    df = pd.DataFrame(res).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df = df.rename(columns={0: 'accuracy', 1: 'precision', 2: 'recall', 3: 'f1', 4: 'roc_auc'})

    trace = go.Table(
        header=dict(values=['<b>Fold', '<b>Accuracy', '<b>Precision', '<b>Recall', '<b>F1 score', '<b>Roc auc'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    align=['center'],
                    font=dict(size=15)),
        cells=dict(values=[('1', '2', '3', '4', '5', 'mean', 'std'),
                           np.round(df['accuracy'], 3),
                           np.round(df['precision'], 3),
                           np.round(df['recall'], 3),
                           np.round(df['f1'], 3),
                           np.round(df['roc_auc'], 3)],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   align=['center'], font=dict(size=15)))

    layout = dict(width=800, height=400, title='<b>Cross Validation - 5 folds</b><br>' + subtitle, font=dict(size=15))
    fig = dict(data=[trace], layout=layout)

    py.iplot(fig, filename='styled_table')
