import os
import warnings
from multiprocessing import set_start_method
from src.table15.configs import Configs

import utils.pipeline_utils as plutils
from utils.models_container import ModelsContainer
from utils.data_tables import DataTables


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

    print('This is Version: 0.0.26')

    configs = Configs(configs_path)
    models_configs_paths = configs.get_from_configs('MODEL_CONFIGS_PATHS', param_type='MODELS')
    
    use_feature_importance_scaling = configs.get_from_configs('USE_FEATURE_IMPORTANCE_SCALING', param_type='MODELS')
    ensemble_configs_path = configs.get_from_configs('ENSEMBLE_CONFIGS_PATH', param_type='MODELS')
    
    data_tables = DataTables() \
        .generate_data(configs)

    # Generate and train models
    models_container = ModelsContainer() \
        .populate_data_tables(data_tables.x_train_p, data_tables.y_train_p, data_tables.x_validation_p) \
        .load_models(models_configs_paths, ensemble_configs_path=ensemble_configs_path) \
        .train_models() \
        .store_feature_importance_from_models(use_feature_importance_scaling=use_feature_importance_scaling)

    df_logits_out_by_feature_types = []
    all_joined_dfs_by_feature_types = []
    feature_types = ["numerical", "binary", "categorical", "grouped"]
    for feature_type in feature_types:
        df_logits_out, all_joined_dfs = plutils.generate_table_by_feature_type(configs, data_tables, models_container, feature_type=feature_type)
        df_logits_out_by_feature_types.append(df_logits_out)
        all_joined_dfs_by_feature_types.append(all_joined_dfs)
    
    for df in df_logits_out_by_feature_types:
        if df is not None:
            print(df.head(20))
            
    return df_logits_out_by_feature_types, all_joined_dfs_by_feature_types


if __name__ == '__main__':
    
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/linear.yaml"
    
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/pima.yaml"
    config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/stroke.yaml"
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/synth_data.yaml"
    if config_path:
        df_logits_out, all_joined_dfs = run(configs_path=config_path)
    else:
        df_logits_out, all_joined_dfs = run()

    print('Done!')
