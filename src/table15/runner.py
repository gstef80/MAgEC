import os
import warnings
from multiprocessing import set_start_method

import utils.pipeline_utils as plutils
from utils.model_utils import ModelUtils
from utils.data_utils import DataUtils


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

    print('This is Version: 0.0.24')

    configs = plutils.yaml_parser(configs_path)
    models = plutils.get_from_configs(configs, 'MODELS', param_type='CONFIGS')
    use_ensemble = plutils.get_from_configs(configs, 'USE_ENSEMBLE', param_type='MODELS')
    
    dutils = DataUtils().generate_data(configs)

    # Train models
    print('Training models ...')
    mutils = ModelUtils(dutils.x_train_p, dutils.y_train_p, dutils.x_validation_p)
    models_dict = mutils.train_models(models)
    model_feat_imp_dict = mutils.extract_feature_importance_from_models(models_dict)
    print(f'Finished generating models {list(models_dict.keys())}')

    df_logits_out_num, all_joined_dfs_num = plutils.generate_table_by_feature_type(
        configs, dutils.x_validation_p, dutils.y_validation_p, models_dict, model_feat_imp_dict, dutils.set_feature_values, dutils.validation_stats_dict, 
        dutils.numerical_features, feature_type='numerical')
    
    df_logits_out_bin, all_joined_dfs_bin = plutils.generate_table_by_feature_type(
        configs, dutils.x_validation_p, dutils.y_validation_p, models_dict, model_feat_imp_dict, dutils.set_feature_values, dutils.validation_stats_dict, 
        dutils.binary_features, feature_type='binary')
    
    df_logits_out_cat, all_joined_dfs_cat = plutils.generate_table_by_feature_type(
        configs, dutils.x_validation_p, dutils.y_validation_p, models_dict, model_feat_imp_dict, dutils.set_feature_values, dutils.validation_stats_dict, 
        dutils.categorical_features, feature_type='categorical')
    
    if df_logits_out_num is not None:
        print(df_logits_out_num.head(20))
    if df_logits_out_bin is not None:
        print(df_logits_out_bin.head(20))
    if df_logits_out_cat is not None:
        print(df_logits_out_cat.head(20))
    return [df_logits_out_num, df_logits_out_bin, df_logits_out_cat], [all_joined_dfs_num, all_joined_dfs_bin, all_joined_dfs_cat]


if __name__ == '__main__':
    # config_path = sys.argv[1]
    config_path = '/Users/ag46548/tmp/t15_configs/t15_stroke.yaml'
    # config_path = '/Users/ag46548/tmp/t15_configs/t15_diabs.yaml'
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pima_diabetes.yaml"
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/synth_data_configs.yaml"
    if config_path:
        df_logits_out, all_joined_dfs = run(configs_path=config_path)
    else:
        df_logits_out, all_joined_dfs = run()

    print('Done!')
