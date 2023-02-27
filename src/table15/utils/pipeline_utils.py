import multiprocessing as mp
from collections import defaultdict
from typing import Any, Dict


import numpy as np
import pandas as pd
from src.table15.configs import DataConfigs
from src.table15.utils.data_tables import DataTables

from src.table15.utils.models_container import ModelsContainer


# from table15.utils import magec_utils as mg
from . import magec_utils as mg


def generate_perturbation_predictions(data_tables, perturbation_params, models_dict, mp_manager=None):
    is_multi_process = False
    run_dfs = dict()
    if mp_manager is not None:
        is_multi_process = True
        run_dfs = mp_manager.dict()
        processes = []
    
    keys = []
    baselines = perturbation_params["baselines"]
    for baseline in baselines:
        perturbation_params["baseline"] = baseline
        for model_name in models_dict.keys():
            perturbation_params["model_name"] = model_name
            key = model_name + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model_name + '_0'
            keys.append(key)
            clf = models_dict[model_name]
            if is_multi_process is False:
                if model_name in ['lstm']:
                    clf = clf.model
                run_dfs[key] = run_magecs_single_process(clf, data_tables, key, perturbation_params.copy())
            elif is_multi_process is True:
                p = mp.Process(name=key, target=run_magecs_multiprocess, 
                    args=(run_dfs, clf, data_tables, perturbation_params.copy()))
                processes.append(p)
        
    if is_multi_process:
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    baseline_runs = store_run_dfs_by_baseline(run_dfs, keys)
    return baseline_runs


def run_magecs_single_process(clf, data_tables, key, perturbation_params):
    print('Starting single-process:', key)
    magecs = run_magecs(key, clf, data_tables, perturbation_params)
    return magecs
    

def run_magecs_multiprocess(return_dict, clf, data_tables, perturbation_params):
    p_name = mp.current_process().name
    print('Starting multi-process:', p_name)
    magecs = run_magecs(p_name, clf, data_tables, perturbation_params)
    return_dict[p_name] = magecs
    

def run_magecs(name, clf, data_tables, perturbation_params):
    magecs = mg.case_magecs(clf, data_tables.x_test, perturbation_params, data_tables.setted_numerical_values)
    print('Magecs for {} computed...'.format(name))
    # magecs = mg.normalize_magecs(magecs, features=perturbation_params["features"], model_name=perturbation_params["model_name"])
    # print('Magecs for {} normalized...'.format(name))
    magecs = magecs.merge(data_tables.Y_test, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', name)
    return magecs


def combine_baseline_runs(main_dict, to_combine_dict, baselines):
    for baseline in baselines:
        main_dict[baseline].extend(to_combine_dict[baseline])
    return main_dict


def aggregate_scores(model_runs_per_baseline: pd.DataFrame, model_names: list, features: list):
    
    feats_to_agg_cols = [[mg.create_magec_col(m, f) for m in model_names] for f in features]
    agg_series_list = []
    for cols, feat in zip(feats_to_agg_cols, features):
        agg_series = pd.Series(model_runs_per_baseline[cols].mean(axis=1), name=feat)
        # agg_series = np.exp(agg_series)
        agg_series_list.append(agg_series)
    return pd.concat(agg_series_list, axis=1)
    

def score_models_per_baseline(baseline_runs, data_tables, models_container, features, use_rank=False):
    baseline_to_scores_df = {}
    all_joined_dfs = {}
    model_names = list(models_container.models_dict.keys())
    for baseline, model_runs in baseline_runs.items():
        model_runs_per_baseline = mg.magec_models(*model_runs,
                            Xdata=data_tables.x_test,
                            Ydata=data_tables.Y_test,
                            features=features)
        if use_rank is False:
            scores_df = aggregate_scores(model_runs_per_baseline, model_names, features)
        else:
            baseline_ranked_df = mg.magec_rank(model_runs_per_baseline, 
                                            rank=len(features), features=features, models=model_names)
            scores_df = agg_scores(baseline_ranked_df, models_container.model_feat_imp_dict, model_names)

        all_joined_dfs[baseline] = model_runs_per_baseline
        baseline_to_scores_df[baseline] = scores_df
    return baseline_to_scores_df, all_joined_dfs


def agg_scores(ranked_df, model_feat_imp_dict, models):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, model_feat_imp_dict, use_weights=False)
        out.append(scores)
    
    return pd.DataFrame.from_records(out)


def get_string_repr(df, feats):
    base_strings = []
    if not df.empty:
        for feat in feats:
            mean = round(df[feat].mean(), 4)
            # std = round(df[feat].std(), 4)
            sem = round(df[feat].sem(), 4)
            # string_repr = f'{mean} +/- {std}'
            string_repr = f'{mean:.3f} ({sem:.3f})'
            base_strings.append(string_repr)
    return base_strings


def produce_output_df(output, features, baselines, test_stats_dict):
    df_out = pd.DataFrame.from_records(output)
    df_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    # df_out = df_out.rename(columns={'0': 'full'})
    df_out = df_out[cols]
    
    if test_stats_dict is not None:
        for stat, stats_series in test_stats_dict.items():
            df_out[stat] = stats_series.values
    
    return df_out


def visualize_output(baseline_to_scores_df, baselines, features, test_stats_dict_feature_type):
    output = {}
    for baseline in baselines:
        df_out = pd.DataFrame.from_records(baseline_to_scores_df[baseline])
        output[baseline] = get_string_repr(df_out, features)
    
    # # TODO: fix baselines upstream  to handle None as 0
    # formatted_baselines = baselines.copy()

    df_out =  produce_output_df(output, features, baselines, test_stats_dict_feature_type)
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


def generate_table_by_feature_type(data_tables: DataTables, models_container: ModelsContainer, 
                                   feature_type: str='numerical', use_multiprocessing: bool=True):
    print(f'Generating Table1.5 for {feature_type} features')
    
    features = data_tables.get_features_by_type(feature_type)
    if features is None or len(features) == 0:
        return None, None

    data_configs = data_tables.data_configs
    if feature_type in ["numerical", "grouped"]:
        configs_key = f"{feature_type.upper()}_INTENSITIES"
        perturbation_intensities = data_configs.get_from_configs(configs_key, param_type="PERTURBATIONS", 
                                                                 default=[1., 0.5, 0.1])
    elif feature_type == 'binary':
        perturbation_intensities = [0, 1]
    elif feature_type == 'categorical':
        perturbation_intensities = [1]
        
    output_type = data_configs.get_from_configs("OUTPUT_TYPE", param_type="PERTURBATIONS")
        
    perturbation_params = {
        "baselines": perturbation_intensities,
        "features": features,
        "feature_type": feature_type,
        "output_type": output_type
    }

    if use_multiprocessing is True:
        baseline_runs = baseline_runs_via_multip(data_tables, models_container, perturbation_params)
    else:
        print('getting magecs for all models with single-processing ...')
        baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, models_container.models_dict)
    if isinstance(features[0], list):
        features = ["::".join(group) for group in data_tables.grouped_features]
    baseline_to_scores_df, all_joined_dfs = score_models_per_baseline(baseline_runs, data_tables, models_container, features)

    df_logits_out = visualize_output(baseline_to_scores_df, perturbation_intensities, features, data_tables.test_stats_dict.get(feature_type))

    return df_logits_out, all_joined_dfs


def baseline_runs_via_multip(data_tables: DataTables, models_container: ModelsContainer, perturbation_params: Dict[str, Any]):
    # Flag for single-process models
    has_tf_models = False
    if 'mlp' in models_container.models_dict:
        has_tf_models = True

    mp_models_dict = models_container.models_dict.copy()
    if has_tf_models:
        tf_models_list = ['mlp']
        # if use_ensemble is True:
        #     tf_models_list.append('ensemble')
        tf_models_dict = {tf_model: models_container.models_dict[tf_model] for tf_model in tf_models_list}
        for tf_model in tf_models_list:
            del mp_models_dict[tf_model]

    with mp.Manager() as manager:
        print('getting magecs for non-TF models via multiprocessing...')
        baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, mp_models_dict, mp_manager=manager)
        print('Done multiprocessing')
    
    if has_tf_models:
        print('getting magecs for TF models with single-processing ...')
        tf_baseline_runs = generate_perturbation_predictions(
            data_tables, perturbation_params, tf_models_dict, mp_manager=None)
        baselines = perturbation_params["baselines"]
        baseline_runs = combine_baseline_runs(baseline_runs, tf_baseline_runs, baselines)
    
    return baseline_runs
