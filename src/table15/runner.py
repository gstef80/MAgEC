from email.mime import base
import pandas as pd
import multiprocessing as mp
from multiprocessing import set_start_method
from . import magec_utils as mg
from . import pima_utils as pm
from . import pipeline_utils as plutils
import time
from collections import defaultdict
import warnings

def run(configs_path='../configs/pima_diabetes.yaml'):
    warnings.filterwarnings('ignore')
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    # TODO: adjust spawn method to start WITH multiprocessing. Most likely with mp.Pool()

    print('This is Version: 0.0.8')

    configs = plutils.yaml_parser(configs_path)
    baselines = plutils.get_from_configs(configs, 'BASELINES', param_type='CONFIGS')
    models = plutils.get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    policy = plutils.get_from_configs(configs, 'POLICY', param_type='CONFIGS')

    df, features, x_train_p, x_validation_p, y_train_p, y_validation_p = plutils.generate_data(configs)
    print('x_train.shape:', x_train_p.shape)
    print('y_train.shape:', y_train_p.shape)

    # Train models
    models_dict = plutils.train_models(x_train_p, y_train_p, models, configs)
    
    # Format check for Yaml configs
    if baselines is None:
        baselines = [None]

    has_tf_models = False
    if 'mlp' in models_dict:
        has_tf_models = True

    sk_models_dict = models_dict.copy()
    if has_tf_models:
        tf_models_list = ['mlp', 'ensemble']
        tf_models = {tf_model: models_dict[tf_model] for tf_model in tf_models_list}
        for tf_model in tf_models_list:
            del sk_models_dict[tf_model]

    print('getting magecs...')
    with mp.Manager() as manager:
        run_dfs = manager.dict()
        processes = []
        keys = []
        for baseline in baselines:
            for model in sk_models_dict.keys():
                key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
                keys.append(key)
                clf = sk_models_dict[model]
                if model in ['mlp', 'lstm']:
                    clf = clf.model
                p = mp.Process(name=key, target=plutils.run_magecs_multip, 
                    args=(run_dfs, clf, x_validation_p, y_validation_p, model, baseline, features))
                processes.append(p)
    
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # TODO: Def this process:
        baseline_runs = defaultdict(list)
        for key in keys:
            baseline = key.split('_')[1]
            if baseline[0] == 'p':
                baseline = int(baseline[1:]) / 100
            else:
                baseline = int(baseline)
            yaml_check = baseline
            if baseline == 0:
                yaml_check = None
            assert yaml_check in baselines
            baseline_runs[baseline].append(run_dfs[key])

    if has_tf_models:
        # TODO: fix multiprocessing for tensorflow based models
        tf_run_dfs = dict()
        keys = []
        for baseline in baselines:
            for model in tf_models.keys():
                key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
                keys.append(key)
                clf = models_dict[model]
                if model in ['mlp', 'lstm']:
                    clf = clf.model
                tf_run_dfs[key] = plutils.run_magecs_single(clf, x_validation_p, y_validation_p, model, key, baseline, features)
        # TODO: Def this process:
        tf_baseline_runs = defaultdict(list)
        for key in keys:
            baseline = key.split('_')[1]
            if baseline[0] == 'p':
                baseline = int(baseline[1:]) / 100
            else:
                baseline = int(baseline)
            yaml_check = baseline
            if baseline == 0:
                yaml_check = None
            assert yaml_check in baselines
            tf_baseline_runs[baseline].append(tf_run_dfs[key])

        for baseline in baselines:
            if baseline is None:
                baseline = 0
            baseline_runs[baseline].extend(tf_baseline_runs[baseline])

    # TODO: Def this process:
    baseline_to_scores_df = {}
    all_joined = {}
    for baseline, model_runs in baseline_runs.items():
        baseline_joined = mg.magec_models(*model_runs,
                            Xdata=x_validation_p,
                            Ydata=y_validation_p,
                            features=features)
        baseline_ranked_df = mg.magec_rank(baseline_joined, rank=len(features), features=features, models=models)
        scores_df = agg_scores(baseline_ranked_df, policy=policy, models=models)

        all_joined[baseline] = baseline_joined
        baseline_to_scores_df[baseline] = scores_df

    output_logits = {}
    output_probs = {}

    for baseline in baselines:
        if baseline is None:
            baseline = 0
        df_logits = pd.DataFrame.from_records(baseline_to_scores_df[baseline]['logits'])
        df_probs = pd.DataFrame.from_records(baseline_to_scores_df[baseline]['probs'])

        if baseline in [None, 0]:
            baseline = 1.0
        base_logits_strings = get_string_repr(df_logits, features)
        base_probs_strings = get_string_repr(df_probs, features)

        output_logits[baseline] = base_logits_strings
        output_probs[baseline] = base_probs_strings
    
    # TODO: fix baselines upstream  to handle None as 0
    if None in baselines:
        idx = baselines.index(None)
        baselines[idx] = 1.0
    
    df_logits_out = pd.DataFrame.from_records(output_logits)
    df_logits_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    df_logits_out = df_logits_out.rename(columns={'0': 'full'})
    df_logits_out = df_logits_out[cols]

    df_probs_out = pd.DataFrame.from_records(output_probs)
    df_probs_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    df_probs_out = df_probs_out.rename(columns={'0': 'full'})
    df_probs_out = df_probs_out[cols]

    print(df_logits_out.head())
    print(df_probs_out.head())

    return (df_logits_out, df_probs_out), all_joined


def agg_scores(ranked_df, policy='mean', models=('mlp', 'rf', 'lr')):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, use_weights=False, policy=policy)
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
