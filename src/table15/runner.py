import os
import warnings
from multiprocessing import set_start_method

import utils.pipeline_utils as plutils
from utils.data_tables import DataTables
from utils.models_container import ModelsContainer

from src.table15.configs import PipelineConfigs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(configs_path: str='./configs/pima_diabetes.yaml'):
    pipeline_configs = PipelineConfigs(configs_path)

    # TODO: adjust spawn method to start WITH multiprocessing. Most likely with mp.Pool()
    warnings.filterwarnings('ignore')
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    data_configs_path = pipeline_configs.get_from_configs("DATA_CONFIGS_PATH", param_type="DATA")
    models_configs_paths = pipeline_configs.get_from_configs('MODEL_CONFIGS_PATHS', param_type='MODELS')
    use_multiprocessing = pipeline_configs.get_from_configs('USE_MULTIPROCESSING', param_type='DEBUGGING',
                                                            default=True)
    
    
    use_feature_importance_scaling = pipeline_configs.get_from_configs('USE_FEATURE_IMPORTANCE_SCALING', 
                                                                       param_type='MODELS')
    ensemble_configs_path = pipeline_configs.get_from_configs('ENSEMBLE_CONFIGS_PATH', param_type='MODELS')
    
    data_tables = DataTables() \
        .set_data_configs(data_configs_path) \
        .generate_data()

    # Generate and train models
    models_container = ModelsContainer() \
        .populate_data_tables(data_tables.x_train, data_tables.Y_train, data_tables.x_test) \
        .load_models(models_configs_paths, ensemble_configs_path=ensemble_configs_path) \
        .train_models() \
        .store_feature_importance_from_models(use_feature_importance_scaling=use_feature_importance_scaling)

    df_logits_out_by_feature_types = []
    all_joined_dfs_by_feature_types = []
    feature_types = ["numerical", "binary", "categorical", "grouped"]
    for feature_type in feature_types:
        df_logits_out, all_joined_dfs = plutils.generate_table_by_feature_type(data_tables, models_container, 
                                                                               feature_type=feature_type, 
                                                                               use_multiprocessing=use_multiprocessing)
        df_logits_out_by_feature_types.append(df_logits_out)
        all_joined_dfs_by_feature_types.append(all_joined_dfs)
    
    for df in df_logits_out_by_feature_types:
        if df is not None:
            print(df.head(20))
            
    return df_logits_out_by_feature_types, all_joined_dfs_by_feature_types


if __name__ == '__main__':
    
    # config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/pima.yaml"
    config_path = "/Users/ag46548/dev/github/KaleRP/table15/src/table15/configs/pipeline_configs/stroke.yaml"
    
    if config_path:
        df_logits_out, all_joined_dfs = run(configs_path=config_path)
    else:
        df_logits_out, all_joined_dfs = run()

    print('Done!')
    