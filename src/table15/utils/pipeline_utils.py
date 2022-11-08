import multiprocessing as mp
from collections import defaultdict
from nis import cat
from typing import Dict

import numpy as np
import pandas as pd
import shap
import yaml
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def generate_data(configs: Dict):
    def impute(df):
        out = df.copy()
        cols = df.columns
        out[cols] = out[cols].replace(0, np.NaN)
        out[cols] = out[cols].fillna(out[cols].mean())
        return out

    csv_path = get_from_configs(configs, 'CSV_PATH')

    numerical_features = get_from_configs(configs, 'NUMERICAL', param_type='FEATURES')
    categorical_features = get_from_configs(configs, 'CATEGORICAL', param_type='FEATURES')
    binary_features = get_from_configs(configs, 'BINARY', param_type='FEATURES')
    target_feature = get_from_configs(configs, 'TARGET', param_type='FEATURES')
    
    set_feature_values = get_from_configs(configs, 'SET_FEATURE_VALUES', param_type='FEATURES')

    random_seed = get_from_configs(configs, 'RANDOM_SEED', param_type='HYPERPARAMS')
    test_size = get_from_configs(configs, 'TEST_SIZE', param_type='HYPERPARAMS')

    df = pd.read_csv(csv_path)

    if random_seed is not None:
        np.random.seed(random_seed)

    x_num = df.loc[:, numerical_features]
    x_num = impute(x_num)

    x_bin = df.loc[:, binary_features]

    x_cat = df.loc[:, categorical_features].fillna('')
    if not x_cat.empty:
        x_cat = pd.get_dummies(x_cat)

    non_numerical_features = binary_features + list(x_cat.columns)
    features = numerical_features + non_numerical_features

    x = pd.concat([x_num, x_bin, x_cat], axis=1)

    Y = df.loc[:, target_feature]

    x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=test_size, random_state=random_seed)

    stsc = StandardScaler()

    xst_train = stsc.fit_transform(x_train[numerical_features])
    xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=numerical_features)
    xst_train = pd.concat([xst_train, x_train[non_numerical_features]], axis=1)
        
    xst_validation = stsc.transform(x_validation[numerical_features])
    xst_validation = pd.DataFrame(xst_validation, index=x_validation.index, columns=numerical_features)
    xst_validation = pd.concat([xst_validation, x_validation[non_numerical_features]], axis=1)
    
    scaling_df = pd.DataFrame([[None for _ in range(len(numerical_features))]], columns=numerical_features)
    for feat, val in set_feature_values.items():
        scaling_df[feat] = val
    
    scaled_values = dict(zip(numerical_features, stsc.transform(scaling_df[numerical_features]).ravel()))
    for feat, val in set_feature_values.items():
        if feat in numerical_features:
            set_feature_values[feat] = scaled_values[feat]

    # Format
    x_validation_p = xst_validation.copy()
    x_validation_p['timepoint'] = 0
    x_validation_p['case'] = np.arange(len(x_validation_p))
    x_validation_p.set_index(['case', 'timepoint'], inplace=True)
    x_validation_p = x_validation_p.sort_index(axis=1)

    y_validation_p = pd.DataFrame(Y_validation.copy())
    y_validation_p['timepoint'] = 0
    y_validation_p['case'] = np.arange(len(x_validation_p))
    y_validation_p.set_index(['case', 'timepoint'], inplace=True)
    y_validation_p = y_validation_p.sort_index(axis=1)

    # Format
    x_train_p = xst_train.copy()
    x_train_p['timepoint'] = 0
    x_train_p['case'] = np.arange(len(x_train_p))
    x_train_p.set_index(['case', 'timepoint'], inplace=True)
    x_train_p = x_train_p.sort_index(axis=1)

    y_train_p = pd.DataFrame(Y_train.copy())
    y_train_p['timepoint'] = 0
    y_train_p['case'] = np.arange(len(y_train_p))
    y_train_p.set_index(['case', 'timepoint'], inplace=True)
    y_train_p = y_train_p.sort_index(axis=1)

    return df, features, x_train_p, x_validation_p, y_train_p, y_validation_p, set_feature_values


def create_mlp(x_train_p):
    mlp = Sequential()
    mlp.add(Dense(60, input_dim=len(x_train_p.columns), activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(30, input_dim=60, activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1, activation='sigmoid'))
    mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mlp


def train_models(x_train_p, y_train_p, x_test_p, models, use_ensemble=False):
    """
    3 ML models for scaled data
    :param x_train_p:
    :param y_train_p:
    :return:
    """

    estimators = list()
    model_feat_imp_dict = defaultdict(dict)
    features = x_train_p.columns

    if 'lr' in models:
        lr = LogisticRegression(C=1.)
        lr.fit(x_train_p, y_train_p)
        estimators.append(('lr', lr))
        model_feat_imp_dict['lr'] = dict(zip(features, lr.coef_.ravel()))

    if 'rf' in models:
        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(x_train_p, y_train_p)
        sigmoidRF = CalibratedClassifierCV(RandomForestClassifier(n_estimators=1000), cv=5, method='sigmoid')
        sigmoidRF.fit(x_train_p, y_train_p.values)
        estimators.append(('rf', sigmoidRF))
        model_feat_imp_dict['rf'] = dict(zip(features, rf.feature_importances_))

    if 'mlp' in models:
        # mlp = KerasClassifier(build_fn=create_mlp, x_train_p=x_train_p, epochs=100, batch_size=64, verbose=0)
        mlp = create_mlp(x_train_p)
        mlp._estimator_type = "classifier"
        mlp.fit(x_train_p, y_train_p, epochs=100, batch_size=64, verbose=0)
        model_feat_imp_dict['mlp'] = dict(zip(features, get_shap_values(mlp, x_train_p, x_test_p).ravel()))
        estimators.append(('mlp', mlp))
        print(model_feat_imp_dict['mlp'])
    
    # Seems to be an issue using KerasClassifier (for ensemble) with a pretrained model when calling predict downstream
    if use_ensemble:
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting='soft')
        ensemble._estimator_type = "classifier"
        ensemble.fit(x_train_p, y_train_p)
        estimators.append(('ensemble', ensemble))
    
    models_dict = dict()
    for model_name, clf in estimators:
        models_dict[model_name] = clf
    
    return models_dict, model_feat_imp_dict

def get_shap_values(model, x_train, x_test):
    background = x_train.to_numpy()#[np.random.choice(x_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_test.to_numpy())
    shap_means = np.mean(shap_values, axis=1)
    l2_norm = np.linalg.norm(shap_means)
    normalized_shap_means = shap_means / l2_norm
    return normalized_shap_means


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


def score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, model_feat_imp_dict, policy):
    baseline_to_scores_df = {}
    all_joined_dfs = {}
    for baseline, model_runs in baseline_runs.items():
        baseline_joined = mg.magec_models(*model_runs,
                            Xdata=x_validation_p,
                            Ydata=y_validation_p,
                            features=features)
        baseline_ranked_df = mg.magec_rank(baseline_joined, rank=len(features), features=features, models=models)
        scores_df = agg_scores(baseline_ranked_df, model_feat_imp_dict, policy=policy, models=models)

        all_joined_dfs[baseline] = baseline_joined
        baseline_to_scores_df[baseline] = scores_df
    return baseline_to_scores_df, all_joined_dfs


def agg_scores(ranked_df, model_feat_imp_dict, policy='mean', models=('mlp', 'rf', 'lr')):
    cols = list(set(ranked_df.columns) - {'case', 'timepoint', 'Outcome'})
    magecs_feats = mg.name_matching(cols, models)
    out = list()
    for (idx, row) in ranked_df.iterrows():
        scores = mg.magec_scores(magecs_feats, row, model_feat_imp_dict, use_weights=False, policy=policy)
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


def produce_output_df(output, features, baselines):
    df_out = pd.DataFrame.from_records(output)
    df_out['feature'] = features
    # re-order cols
    cols = ['feature'] + baselines
    df_out = df_out.rename(columns={'0': 'full'})
    df_out = df_out[cols]
    return df_out


def visualize_output(baseline_to_scores_df, baselines, features):
    output = {}
    for baseline in baselines:
        df_out = pd.DataFrame.from_records(baseline_to_scores_df[baseline])
        output[baseline] = get_string_repr(df_out, features)
    
    # TODO: fix baselines upstream  to handle None as 0
    formatted_baselines = baselines.copy()

    df_out =  produce_output_df(output, features, formatted_baselines)
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
                                   feature_type='numerical'):
    print(f'Generating Table1.5 for {feature_type} features')

    models = get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    use_ensemble = get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')
    policy = get_from_configs(configs, 'POLICY', param_type='CONFIGS')
    skip_multiprocessing = get_from_configs(configs, 'SKIP_MULTIPROCESSING', param_type='MODELS')

    if feature_type == 'numerical':
        features = get_from_configs(configs, 'NUMERICAL', param_type='FEATURES')
        baselines = get_from_configs(configs, 'BASELINES', param_type='CONFIGS')
        if len(features) == 0:
            return None, None
    elif feature_type == 'binary':
        features = get_from_configs(configs, 'BINARY', param_type='FEATURES')
        baselines = [0, 1]
        if len(features) == 0:
            return None, None
    # elif feature_type == 'categorical':
    #     features = get_from_configs(configs, 'CATEGORICAL', param_type='FEATURES')
    #     # baselines = 
    else:
        raise ValueError('Feature type must be numerical, binary, or categorical')

    if skip_multiprocessing is False:
        baseline_runs = baseline_runs_via_multip(
            models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, use_ensemble=use_ensemble)
        
    else:
        print('getting magecs for all models with single-processing ...')
        baseline_runs = generate_perturbation_predictions(
            models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, mp_manager=None)

    baseline_to_scores_df, all_joined_dfs = score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, model_feat_imp_dict, policy)

    df_logits_out = visualize_output(baseline_to_scores_df, baselines, features)

    return df_logits_out, all_joined_dfs


def baseline_runs_via_multip(models_dict, x_validation_p, y_validation_p, baselines, features, feature_type, set_feature_values, use_ensemble=True):
    # Flag for single-process models
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
