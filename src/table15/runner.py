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
    set_start_method("spawn")

    print('This is Version: 0.0.7')

    configs = plutils.yaml_parser(configs_path)
    baselines = plutils.get_from_configs(configs, 'BASELINES')
    policy = plutils.get_from_configs(configs, 'POLICY')
    features = plutils.get_from_configs(configs, 'FEATURES')
    models = plutils.get_from_configs(configs, 'MODELS')

    if baselines is None:
        baselines = [None]

    pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p = pm.pima_data(configs)
    print(x_train_p.shape)
    print(y_train_p.shape)

    # Train models
    models_dict = pm.pima_models(x_train_p, y_train_p, models)

    if 'ensemble' in models_dict:
        models.append('ensemble')

    print('getting magecs...')
    with mp.Manager() as manager:
        run_dfs = manager.dict()
        processes = []
        keys = []
        for baseline in baselines:
            for model in models_dict.keys():
                key = model + '_p{}'.format(int(baseline * 100)) if baseline not in [None, 'None'] else model + '_0'
                keys.append(key)
                clf = models_dict[model]
                if model in ['mlp', 'lstm']:
                    clf = clf.model
                # run_magecs(run_dfs, clf, x_validation_p, y_validation_p, model, key, baseline)
                p = mp.Process(name=key,
                                            target=run_magecs, 
                                            args=(run_dfs, clf, x_validation_p, y_validation_p, model, baseline, features))
                processes.append(p)

        for p in processes:
            p.start()
        for process in processes:
            process.join()
        
        # Def this process:
        baseline_runs = defaultdict(list)
        keys = sorted(keys)
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
        
        # Def this process:
        baseline_scores = {}
        all_joined = {}
        for baseline, model_runs in baseline_runs.items():
            baseline_joined = mg.magec_models(*model_runs,
                                Xdata=x_validation_p,
                                Ydata=y_validation_p,
                                features=features)
            baseline_ranked_df = mg.magec_rank(baseline_joined, rank=len(features), features=features, models=models)
            baseline_scores_df = agg_scores(baseline_ranked_df, policy=policy, models=models)

            all_joined[baseline] = baseline_joined
            baseline_scores[baseline] = baseline_scores_df

        return baseline_scores, all_joined


def agg_scores(ranked_df, policy='mean', models=('mlp', 'rf', 'lr')):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, use_weights=False, policy=policy)
        out.append(scores)
    
    return pd.DataFrame.from_records(out)

def     run_magecs(return_dict, clf, x_validation_p, y_validation_p, model_name, baseline=None, features=None):
    print(model_name)
    p_name = mp.current_process().name
    print('Starting:', p_name)
    if model_name == 'lstm':
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline, timeseries=True)
    else:
        magecs = mg.case_magecs(clf, x_validation_p, model_name=model_name, baseline=baseline)
    print('Magecs for {} computed...'.format(p_name))
    magecs = mg.normalize_magecs(magecs, features=features, model_name=model_name)
    print('Magecs for {} normalized...'.format(p_name))
    magecs = magecs.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    print('Exiting :', p_name)
    return_dict[p_name] = magecs
