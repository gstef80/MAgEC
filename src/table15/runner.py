import multiprocessing as mp
import os
import sys
import warnings
from multiprocessing import set_start_method

import table15.utils.pipeline_utils as plutils


def run(configs_path='./configs/pima_diabetes.yaml'):
    if not os.path.isabs(configs_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        configs_path = os.path.join(script_dir, configs_path)

    warnings.filterwarnings('ignore')
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    # TODO: adjust spawn method to start WITH multiprocessing. Most likely with mp.Pool()

    print('This is Version: 0.0.17')

    configs = plutils.yaml_parser(configs_path)
    baselines = plutils.get_from_configs(configs, 'BASELINES', param_type='CONFIGS')
    models = plutils.get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    policy = plutils.get_from_configs(configs, 'POLICY', param_type='CONFIGS')
    use_ensemble = plutils.get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')
    skip_multiprocessing = plutils.get_from_configs(configs, 'SKIP_MULTIPROCESSING', param_type='MODELS')

    df, features, x_train_p, x_validation_p, y_train_p, y_validation_p = plutils.generate_data(configs)

    # Train models
    print('Training models ...')
    models_dict = plutils.train_models(x_train_p, y_train_p, models, use_ensemble=use_ensemble)
    print(f'Finished training models {list(models_dict.keys())}')

    if skip_multiprocessing is False:
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
            print('getting magecs for via multiprocessing...')
            baseline_runs = plutils.generate_perturbation_predictions(
                mp_models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=manager)
            print('Done multiprocessing')
        
        if has_tf_models:
            print('getting magecs for TF model with single-processing ...')
            tf_baseline_runs = plutils.generate_perturbation_predictions(
                tf_models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=None)

            baseline_runs = plutils.combine_baseline_runs(baseline_runs, tf_baseline_runs, baselines)

    else:
        print('getting magecs for all models with single-processing ...')
        baseline_runs = plutils.generate_perturbation_predictions(
            models_dict, x_validation_p, y_validation_p, baselines, features, mp_manager=None)


    baseline_to_scores_df, all_joined_dfs = plutils.score_models_per_baseline(baseline_runs, x_validation_p, y_validation_p, features, models, policy)

    df_logits_out = plutils.visualize_output(baseline_to_scores_df, baselines, features)

    return df_logits_out, all_joined_dfs


if __name__ == '__main__':
    config_path = sys.argv[1]
    if config_path:
        run(configs_path=config_path)
    else:
        run()
