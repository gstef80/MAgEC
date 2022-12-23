import multiprocessing as mp
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import yaml

# from table15.utils import magec_utils as mg
from . import magec_utils as mg


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


def generate_perturbation_predictions(models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, mp_manager=None):
    is_multi_process = False
    run_dfs = dict()
    if mp_manager is not None:
        is_multi_process = True
        run_dfs = mp_manager.dict()
        processes = []
    
    keys = []
    for baseline in baselines:
        for model in models_dict.keys():
            key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
            keys.append(key)
            clf = models_dict[model]
            if is_multi_process is False:
                if model in ['lstm']:
                    clf = clf.model
                run_dfs[key] = run_magecs_single(clf, x_validation_p, y_validation_p, model, key, baseline, features, feature_type, set_feature_values)
            elif is_multi_process is True:
                p = mp.Process(name=key, target=run_magecs_multip, 
                    args=(run_dfs, clf, x_validation_p, y_validation_p, model, baseline, features, feature_type, set_feature_values))
                processes.append(p)
        
    if is_multi_process:
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    baseline_runs = store_run_dfs_by_baseline(run_dfs, keys)
    return baseline_runs


def run_magecs_single(clf, x_validation_p, y_validation_p, model_name, key, baseline, features, feature_type, set_feature_values):
    print('Starting single-process:', key)
    is_timeseries = False
    if model_name == 'lstm':
        is_timeseries = True
    magecs = mg.case_magecs(clf, x_validation_p, features, feature_type, set_feature_values, model_name=model_name, baseline=baseline, timeseries=is_timeseries)
    print('Magecs for {} computed...'.format(key))
    magecs = mg.normalize_magecs(magecs, features=features, model_name=model_name)
    print('Magecs for {} normalized...'.format(key))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', key)
    return magecs
    

def run_magecs_multip(return_dict, clf, x_validation_p, y_validation_p, model_name, baseline, features, feature_type, set_feature_values):
    p_name = mp.current_process().name
    print('Starting multi-process:', p_name)
    is_timeseries = False
    if model_name == 'lstm':
        is_timeseries = True
    magecs = mg.case_magecs(clf, x_validation_p, features, feature_type, set_feature_values, model_name=model_name, baseline=baseline, timeseries=is_timeseries)
    print('Magecs for {} computed...'.format(p_name))
    magecs = mg.normalize_magecs(magecs, features=features, model_name=model_name)
    print('Magecs for {} normalized...'.format(p_name))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', p_name)
    return_dict[p_name] = magecs


def combine_baseline_runs(main_dict, to_combine_dict, baselines):
    for baseline in baselines:
        main_dict[baseline].extend(to_combine_dict[baseline])
    return main_dict


def score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, model_feat_imp_dict, policy, num_models_rank):
    baseline_to_scores_df = {}
    all_joined_dfs = {}
    for baseline, model_runs in baseline_runs.items():
        baseline_joined = mg.magec_models(*model_runs,
                            Xdata=x_validation_p,
                            Ydata=y_validation_p,
                            features=features)
        baseline_ranked_df = mg.magec_rank(baseline_joined, rank=len(features), features=features, models=models)
        scores_df = agg_scores(baseline_ranked_df, model_feat_imp_dict, policy=policy, models=models, num_models_rank=num_models_rank)

        all_joined_dfs[baseline] = baseline_joined
        baseline_to_scores_df[baseline] = scores_df
    return baseline_to_scores_df, all_joined_dfs


def agg_scores(ranked_df, model_feat_imp_dict, policy='mean', models=('mlp', 'rf', 'lr'), num_models_rank=None):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, model_feat_imp_dict, use_weights=False,
                                 policy=policy, num_models_rank=num_models_rank)
        out.append(scores)
    
    return pd.DataFrame.from_records(out)


def get_string_repr(df, feats):
    base_strings = []
    for feat in feats:
        mean = round(df[feat].mean(), 4)
        # std = round(df[feat].std(), 4)
        sem = round(df[feat].sem(), 4)
        # string_repr = f'{mean} +/- {std}'
        string_repr = f'{mean} ({sem})'
        base_strings.append(string_repr)
    return base_strings


def produce_output_df(output, features, baselines, validation_stats_dict):
    df_out = pd.DataFrame.from_records(output)
    df_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    # df_out = df_out.rename(columns={'0': 'full'})
    df_out = df_out[cols]
    
    # for stat, stats_series in validation_stats_dict.items():
    #     df_out[stat] = stats_series.values
    
    return df_out


def visualize_output(baseline_to_scores_df, baselines, features, validation_stats_dict):
    output = {}
    for baseline in baselines:
        df_out = pd.DataFrame.from_records(baseline_to_scores_df[baseline])
        output[baseline] = get_string_repr(df_out, features)
    
    # TODO: fix baselines upstream  to handle None as 0
    formatted_baselines = baselines.copy()

    df_out =  produce_output_df(output, features, formatted_baselines, validation_stats_dict)
    return df_out


def store_run_dfs_by_baseline(run_dfs, keys):
    baseline_runs = defaultdict(list)
    for key in keys:
        baseline = key.split('_')[1]
        if baseline[0] == 'p':
            baseline = int(baseline[1:]) / 100
        else:
            baseline = int(baseline)
        baseline_runs[baseline].append(run_dfs[key])
    return baseline_runs


def generate_table_by_feature_type(configs, x_validation_p, y_validation_p, models_dict, model_feat_imp_dict, set_feature_values,
                                   validation_stats_dict, features, feature_type='numerical'):
    print(f'Generating Table1.5 for {feature_type} features')

    models = get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    use_ensemble = get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')
    policy = get_from_configs(configs, 'POLICY', param_type='CONFIGS')
    skip_multiprocessing = get_from_configs(configs, 'SKIP_MULTIPROCESSING', param_type='MODELS')
    num_models_rank = get_from_configs(configs, 'NUM_MODELS_RANK', param_type='MODELS')
    
    if feature_type not in ["numerical", "binary", "categorical"]:
        raise ValueError('Feature type must be numerical, binary, or categorical')
    
    if len(features) == 0:
        return None, None
    
    if feature_type == 'numerical':
        baselines = get_from_configs(configs, 'BASELINES', param_type='CONFIGS')
    elif feature_type == 'binary':
        baselines = [0, 1]
    elif feature_type == 'categorical':
        baselines = [1]

    if skip_multiprocessing is False:
        baseline_runs = baseline_runs_via_multip(
            models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, use_ensemble=use_ensemble)
        
    else:
        print('getting magecs for all models with single-processing ...')
        baseline_runs = generate_perturbation_predictions(
            models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, mp_manager=None)

    baseline_to_scores_df, all_joined_dfs = score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, model_feat_imp_dict, 
                                                                      policy, num_models_rank)

    df_logits_out = visualize_output(baseline_to_scores_df, baselines, features, validation_stats_dict)

    return df_logits_out, all_joined_dfs


def baseline_runs_via_multip(models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, use_ensemble=True):
    # Flag for single-process models
    has_tf_models = False
    if 'mlp' in models_dict:
        has_tf_models = True

    mp_models_dict = models_dict.copy()
    if has_tf_models:
        tf_models_list = ['mlp']
        if use_ensemble is True:
            tf_models_list.append('ensemble')
        tf_models_dict = {tf_model: models_dict[tf_model] for tf_model in tf_models_list}
        for tf_model in tf_models_list:
            del mp_models_dict[tf_model]

    with mp.Manager() as manager:
        print('getting magecs for non-TF models via multiprocessing...')
        baseline_runs = generate_perturbation_predictions(
            mp_models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, mp_manager=manager)
        print('Done multiprocessing')
    
    if has_tf_models:
        print('getting magecs for TF models with single-processing ...')
        tf_baseline_runs = generate_perturbation_predictions(
            tf_models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, mp_manager=None)

        baseline_runs = combine_baseline_runs(baseline_runs, tf_baseline_runs, baselines)
    
    return baseline_runs
