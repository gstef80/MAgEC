import heapq
from collections import OrderedDict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import rbo
import seaborn as sns
from matplotlib.font_manager import FontProperties
from numpy import interp
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score


def get_logit_base2(prob, eps=1e-16):
    return np.log2((prob+eps)/(1-prob+eps))


def get_logit_ln(prob, eps=1e-16):
    return np.log((prob+eps)/(1-prob+eps))


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


def zero_pad(df, length=None):
    """
    Given a MAgEC dataframe (indexed with timepoint and case) zero-pad all features/columns
    """
    x = list()
    y = list()
    z = list()

    # assert 'timepoint' and 'case' exist in either index or columns
    assert 'timepoint' in df.index.names, "mising 'timepoint' from index"
    assert 'case' in df.index.names, "mising 'case' from index"

    if length is None:
        length = len(df.index.get_level_values('timepoint').unique())

    # use all features except for 'label', 'case' and 'timepoint'
    series_cols = list(set(df.columns) - {'case', 'timepoint', 'label'})

    for idx, fname in df.groupby(level='case', group_keys=False):

        if 'label' in df.columns:
            y_data = np.array(fname['label'].values[0])
            y.append(y_data)

        tmp = fname[series_cols].astype(float).values  # get all features as matrix of floats
        x_data = np.zeros([length, tmp.shape[1]])  # prepare zero pad matrix
        x_data[:tmp.shape[0], :] = tmp  # zero pad
        x.append(x_data)
        # format for pandas dataframe with columns containg time-series
        series = [[x_data[i, j] for i in range(x_data.shape[0])] for j in range(x_data.shape[1])]

        if 'label' in df.columns:
            z.append(pd.Series(series + [idx, y_data],
                               index=series_cols + ['case', 'label']))
        else:
            z.append(pd.Series(series + [idx],
                               index=series_cols + ['case']))
    x = np.array(x)
    y = np.array(y)
    z = pd.DataFrame.from_records(z)

    return x, y, z


def slice_series(target_data, tt, reverse=True):
    if reverse:
        df = target_data.loc[target_data.index.get_level_values('timepoint') >= tt]
    else:
        df = target_data.loc[target_data.index.get_level_values('timepoint') <= tt]
    return df


def static_prediction(model, target_data, set_feature_values, score_preprocessing,
                      timepoint, var_name, epsilons, label='orig', baseline=1.0):
    idx = target_data.index.get_level_values('timepoint') == timepoint
    if label == 'orig':
        df = target_data.loc[idx].copy()
    elif label == 'perturb':
        # perturb to baseline conditions
        df = target_data.loc[idx].copy()
        if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
            new_val = baseline
            # switch binary values
            # new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
            # new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
        elif type(epsilons[var_name]) is list:
            raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
        else:
            if var_name in set_feature_values:
                set_val = set_feature_values[var_name]
            else:
                set_val = 0
            curr_val = df.loc[:, var_name]
            pert_dist = curr_val - set_val
            new_val = curr_val - (pert_dist * float(baseline))
        df.loc[:, var_name] = new_val
    else:
        raise ValueError("label must be either 'orig' or' 'perturb")
    probs = predict(model, df)
    logits = score_preprocessing(probs)
    df_cols = df.columns
    df['probs_{}'.format(label)] = probs
    df['logit_{}'.format(label)] = logits
    df = df.drop(df_cols, axis=1)
    return df


def series_prediction(model, target_data, set_feature_values, score_preprocessing,
                      timepoint, reverse, pad, var_name, epsilons,
                      label='orig', baseline=1.0):
    if label == 'orig':
        df = target_data.copy()
    elif label == 'perturb':
        df = target_data.copy()
        idx = df.index.get_level_values('timepoint') == timepoint
        if baseline in [None, 'None']:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                new_val = epsilons[var_name]
            df.loc[idx, var_name] = new_val  # perturb to new value
        else:
            if type(epsilons[var_name]) is list and len(epsilons[var_name]) == 2:
                new_val = (df.loc[:, var_name] == epsilons[var_name][0]).astype(int)
                new_val = new_val.multiply(epsilons[var_name][1]) + (1-new_val).multiply(epsilons[var_name][0])
            elif type(epsilons[var_name]) is list:
                raise ValueError('epsilon value can only be a scalar or have 2 values (binary)')
            else:
                tmp = df.loc[:, var_name]
                new_val = tmp - tmp * float(baseline)
            df.loc[idx, var_name] = new_val
    else:
        raise ValueError("label must be either 'orig' or' 'perturb")
    df = slice_series(df, timepoint, reverse=reverse)
    x_series, _, df_vector = zero_pad(df, length=pad)
    df_cols = list(set(df_vector.columns) - {'case'})
    probs = predict(model, x_series)
    logits = score_preprocessing(probs)
    df_vector['probs_{}'.format(label)] = probs
    df_vector['logit_{}'.format(label)] = logits
    df_vector['timepoint'] = timepoint
    df_vector = df_vector.set_index(['case', 'timepoint'])
    df_vector = df_vector.drop(df_cols, axis=1)
    return df_vector.loc[df_vector.index.get_level_values('timepoint') == timepoint]


def z_perturbation(model, target_data, features, feature_type, set_feature_values,
                   score_preprocessing=get_logit_ln,
                   score_comparison=lambda x_baseline, x: x - x_baseline,
                   sort_categories=True,
                   categories=None,
                   binary=None,
                   timepoint_level='timepoint',
                   epsilon_value=0,
                   reverse=True,
                   timeseries=False,
                   baseline=1.0):
    '''
    Main method for computing a MAgEC. Assumes 'scaled/normalized' features in target data.
        Supporting 2 types of variables:
        - numeric / floats
        - binary / boolean
        Default score_comparison subtracts perturbed output from original.
        For a binary classification task, where 1 denotes a "bad" outcome, a good perturbation
        is expected to result in a negative score_comparison (assuming monotonic score_preprocessing).
    :param model:
    :param target_data:
    :param score_preprocessing:
    :param score_comparison:
    :param sort_categories:
    :param categories:
    :param features:
    :param binary:
    :param timepoint_level:
    :param epsilon_value:
    :param reverse:
    :param timeseries:
    :param baseline: whether to compute baseline MAgECS, None as default, 0.01 for 1% perturbation
    :return:
    '''
    # assert 'timepoint' and 'case' exist in either index or columns
    assert 'timepoint' in target_data.index.names, "mssing 'timepoint' from index"
    assert 'case' in target_data.index.names, "missing 'case' from index"

    timepoints = list(sorted(target_data.index.get_level_values(timepoint_level).unique()))
    if reverse:
        timepoints = list(reversed(timepoints))

    assert len(features) > 0, f"No features here to perturb. Feature type: {feature_type}."
    features = np.asarray(features)

    if binary is None:
        binary = target_data[features].apply(lambda x: len(np.unique(x)), ) <= 2
        binary = binary[binary].index.tolist()
    assert (feature_type == 'binary' and len(binary) > 0) or (feature_type != 'binary' and len(binary) == 0), (
        f"Mismatch between binary feature_type = {feature_type} and len(binary) = {len(binary)}")

    epsilons = dict()
    for var_name in features:
        if var_name in binary:
            epsilons[var_name] = target_data[var_name].unique().tolist()
            # epsilons[var_name] = target_data[var_name].value_counts().idxmax()  # most frequent value
        else:
            epsilons[var_name] = epsilon_value

    if categories is None:
        categories = get_column_categories(target_data[features], sort=sort_categories)

    prob_deltas_per_cell = pd.DataFrame(index=target_data.index,
                                        columns=pd.Index(hier_col_name_generator(categories),
                                                         name='features'))

    for tt in timepoints:

        # print("Timepoint {}".format(tt))

        if not timeseries:
            base = static_prediction(model,
                                     target_data,
                                     set_feature_values,
                                     score_preprocessing,
                                     tt,
                                     var_name=None,
                                     epsilons=None,
                                     label='orig')
        else:
            base = series_prediction(model,
                                     target_data,
                                     set_feature_values,
                                     score_preprocessing,
                                     tt,
                                     reverse,
                                     len(timepoints),
                                     var_name=None,
                                     epsilons=None,
                                     label='orig')

        for var_name in features:

            if not timeseries:
                # predict for perturbed data
                perturb = static_prediction(model,
                                            target_data,
                                            set_feature_values,
                                            score_preprocessing,
                                            tt,
                                            var_name=var_name,
                                            epsilons=epsilons,
                                            label='perturb',
                                            baseline=baseline)
            else:
                # predict for perturbed data
                perturb = series_prediction(model,
                                            target_data,
                                            set_feature_values,
                                            score_preprocessing,
                                            tt,
                                            reverse,
                                            len(timepoints),
                                            var_name=var_name,
                                            epsilons=epsilons,
                                            label='perturb',
                                            baseline=baseline)
            # logits
            logit_orig = base['logit_orig']
            logit_perturb = perturb['logit_perturb']
            logit_diff = score_comparison(logit_orig, logit_perturb)
            # store
            idx = target_data.index.get_level_values('timepoint') == tt
            # prob_deltas_per_cell.at[idx, var_name] = logit_diff
            prob_deltas_per_cell.loc[idx, var_name] = logit_diff
            prob_deltas_per_cell.loc[idx, 'perturb_{}_prob'.format(var_name)] = perturb['probs_perturb']
            prob_deltas_per_cell.loc[idx, 'orig_prob'] = base['probs_orig']

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


def create_magec_col(model_name, feature):
    return model_name + '_' + feature


def case_magecs(model, data, features, feature_type, set_feature_values, epsilon_value=0, model_name=None,
                reverse=True, timeseries=False, baseline=1.0, binary=None):
    """
    Compute MAgECs for every 'case' (individual row/member table).
    Use all features in data to compute MAgECs.
    NOTE 1: we prefix MAgECs with model_name.
    NOTE 2: we postfix non-MAgECs, such as 'perturb_<FEAT>_prob' with model_name.
    """
    magecs = z_perturbation(model, data, features, feature_type, set_feature_values,
                            epsilon_value=epsilon_value,
                            reverse=reverse,
                            timeseries=timeseries,
                            baseline=baseline,
                            binary=binary,
                            score_preprocessing=get_logit_ln
                            )

    all_features = magecs.columns
    magecs = magecs.reset_index()
    # rename features in case_magecs to reflect the fact that they are derived for a specific model
    prefix = 'm' if model_name is None else model_name
    postfix = prefix
    for feat in all_features:
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


def magec_cols(magec, features):
    all_cols = magec.columns
    orig_prob_col = [col for col in all_cols if col.startswith('orig_prob_')]
    jcols = ['case', 'timepoint']
    m_cols = [col for col in all_cols if '_'.join(col.split('_')[1:]) in features]
    prob_cols = [col for col in all_cols if col.startswith('perturb_') and
                 col[8:].split('_prob_')[0] in features]
    cols = jcols + m_cols + prob_cols + orig_prob_col
    return jcols, cols


def magec_models(*magecs, **kwargs):
    """
    Wrapper function for joining MAgECs from different models together and (optionally) w/ tabular data
    """
    Xdata = kwargs.get('Xdata', None)
    Ydata = kwargs.get('Ydata', None)
    features = kwargs.get('features', [])
    assert len(magecs) > 1
    jcols, cols = magec_cols(magecs[0], features)
    magec = magecs[0][cols]
    if Xdata is not None:
        magec = magec.merge(Xdata.reset_index(), on=jcols)
    if Ydata is not None:
        magec = magec.merge(Ydata.reset_index(), on=jcols)
    for mgc in magecs[1:]:
        _, cols = magec_cols(mgc, features)
        mgc = mgc[cols]
        magec = magec.merge(mgc, on=jcols)
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
                # if magec < 0:  # negative magecs are originally positive magecs and are filtered out
                #     l.append(None)
                #     l.append("not_found")
                # else:
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
            if feat == 'not_found':
                continue
            # score = scoring(row[score_col])
            score = row[score_col]
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
            scores[feat] /= len(consensus[feat])
        if winner is None or score > winner[1]:
            winner = (feat, score, len(consensus[feat]), sorted(list(consensus[feat])))

    return winner

def magec_scores(magecs_feats,
                 row,
                 model_feat_imp_dict,
                 scoring=lambda w: abs(w),
                 use_weights=False,
                 weights={'rf': None, 'mlp': None, 'lr': None},
                 policy='sum'):
    """
    Returns a dictionary of all MAgEC scores computed as a naive sum across models
    magecs_feats is a dictionary with magec feature column names and magec value column names for every model,
     e.g
    {'rf': {'rf_feat_1': 'rf_magec_1', 'rf_feat_2': 'rf_magec_2'},
     'mlp': {'mlp_feat_1': 'mlp_magec_1', 'mlp_feat_2': 'mlp_magec_2'},
     'lr': {'lr_feat_1': 'lr_magec_1', 'lr_feat_2': 'lr_magec_2'}}
    """
    assert policy in ['sum', 'mean'], "Only 'sum' or 'mean' policy is supported"
    consensus = {}
    scores = {}
    if use_weights:
        assert sorted(weights.keys()) == sorted(magecs_feats.keys())
    for model, feat_dict in magecs_feats.items():
        for feat_col, score_col in feat_dict.items():
            feat = row[feat_col]
            if feat == 'not_found':
                continue

            score = row[score_col]
            
            # Modify score with model feature importance weight
            # model_name = model.split("_")[0]
            feat_imp_weight = model_feat_imp_dict[model][feat]
            score *= feat_imp_weight
            
            # Convert ln(OR) to OR
            score = np.exp(score)

            if score in [None, 'nan']:
                continue
            score = scoring(score)

            if use_weights:
                if weights[model] is not None:
                    score *= weights[model]
            if feat in scores:
                scores[feat] += score
                consensus[feat].append(model)
            else:
                scores[feat] = score
                consensus[feat] = [model]
    if policy == 'mean':
        for feat, score in scores.items():
            scores[feat] = score / len(consensus[feat])
    return scores
