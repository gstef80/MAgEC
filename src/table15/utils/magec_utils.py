import heapq
from collections import OrderedDict

import numpy as np
import pandas as pd
from src.table15.perturbations.group_perturbation import GroupPerturbation

from src.table15.perturbations.z_perturbation import ZPerturbation



def m_prefix(magecs, feature, model_name=None):
    """
    Given a feature (e.g. BMI) and a magecs dataframe extract prefix (model_name).
    """
    prefix = 'm'
    for c in magecs.columns:
        splits = c.split('_')
        if isinstance(feature, list):
            feature = "::".join(feature)
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
    if isinstance(feature, list):
        feature = "::".join(feature)
    return model_name + '_' + feature


def case_magecs(model, data, perturbation_params, set_feature_values):
    """
    Compute MAgECs for every 'case' (individual row/member table).
    Use all features in data to compute MAgECs.
    NOTE 1: we prefix MAgECs with model_name.
    NOTE 2: we postfix non-MAgECs, such as 'perturb_<FEAT>_prob' with model_name.
    """
    features = perturbation_params["features"]
    feature_type = perturbation_params["feature_type"]
    baseline = perturbation_params["baseline"]
    output_type = perturbation_params["output_type"]
    
    if feature_type == "grouped":
        perturbation = GroupPerturbation(data, model, features, feature_type)
    else:
        perturbation = ZPerturbation(data, model, features, feature_type)

    magecs = perturbation.run_perturbation(set_feature_values, output_type, baseline=baseline)
    all_features = magecs.columns
    magecs = magecs.reset_index()
    # rename features in case_magecs to reflect the fact that they are derived for a specific model
    prefix = 'm' if model.name is None else model.name
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
    Output is a Pandas dataframe with computed magecs.
    Note: Positive magecs indicate counter-productive interventions.
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
                neg_magec, feat = heapq.heappop(v[model])
                # if neg_magec < 0:  # negative magecs are originally positive magecs and are filtered out
                if neg_magec <= -1:
                    l.append(None)
                    l.append("not_found")
                else:
                    l.append(-neg_magec)  # retrieve original magec sign
                    l.append(feat)
        out.append(l)

    out = pd.DataFrame.from_records(out)
    # create dataframe's columns
    for model in models:
        if rank == 1:
            columns.append(model + '_magec_1')
            columns.append(model + '_feat_1')
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


def magec_scores(magecs_feats,
                 row,
                 model_feat_imp_dict,
                 scoring=lambda w: abs(w),
                 use_weights=False,
                 weights={'rf': None, 'mlp': None, 'lr': None}):
    """
    Returns a dictionary of all MAgEC scores computed as a naive sum across models
    magecs_feats is a dictionary with magec feature column names and magec value column names for every model,
     e.g
    {'rf': {'rf_feat_1': 'rf_magec_1', 'rf_feat_2': 'rf_magec_2'},
     'mlp': {'mlp_feat_1': 'mlp_magec_1', 'mlp_feat_2': 'mlp_magec_2'},
     'lr': {'lr_feat_1': 'lr_magec_1', 'lr_feat_2': 'lr_magec_2'}}
    """
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
            feat_imp_weight = model_feat_imp_dict[model].get(feat, 1.0)
            score *= feat_imp_weight
            
            # # Convert ln(OR) to OR
            # score = np.exp(score)

            if score in [None, 'nan']:
                continue
            score = scoring(score)

            if use_weights:
                if weights[model] is not None:
                    score *= weights[model]
            if feat in scores:
                scores[feat].append(score)
                consensus[feat].append(model)
            else:
                scores[feat] = [score]
                consensus[feat] = [model]
    for feat in scores.keys():
        # sum the top N (equal to num_models_rank)
        scores[feat] = sum(sorted(scores[feat], reverse=True))#[:num_models_rank])
    # Get mean scores
    for feat, score in scores.items():
        scores[feat] = score / len(consensus[feat])#[:num_models_rank])
    return scores
