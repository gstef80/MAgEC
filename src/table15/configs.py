from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import yaml


class Configs(ABC):
    def __init__(self, file_path_str: str) -> None:
        self.configs_dict: Dict[str, Any]
        self.file_path: Path
        self.configs_dict: Dict[str, Any] = self.import_configs(file_path_str)
        self.assert_compatibility()
        
    @abstractmethod
    def assert_compatibility():
        """
        Abstract method that ensures all Configs classes meet criteria or fail early.
        Defined by implementer.
        """
        pass
        
    def import_configs(self, file_path_str: str) -> Dict[str, Any]:
        """Imports a Yaml configs from file path

        Args:
            file_path_str (str): Yaml file path

        Returns:
            Configs (Dict[str, Any]): Dictionary of user-defined configurations. Can be nested.
        """
        self.file_path = self.to_absolute_path(file_path_str)
        with open(self.file_path, 'r') as file:
            return yaml.safe_load(file)
        
    def get_from_configs(self, key: str, 
                         param_type: str='CONFIGS',
                         default: Any=None,
                         is_assert_compatibility: bool=False
                         ) -> Any:
        """Scans through configurations dict to find key of interest. Optionally, uses param_type to
        specify specific config types so that similar key names can be distinguished.

        Args:
            key (str):  
            param_type (str, optional): First-level keys in Yaml config files for organizational purposes. Defaults to "CONFIGS".
            default (Any, optional): If key is not found, this value is returned. Defaults to None.
            is_assert_compatibility (bool, optional): We ignore Warnings during assertions. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: First value of key found in param_type, and then from overall dict.
            TODO: Loop through all potential keys to find value
        """
        key = key.upper()
        if param_type in self.configs_dict and key in self.configs_dict[param_type]:
            return self.configs_dict[param_type][key]
        if key in self.configs_dict:
            return self.configs_dict[key]
        if is_assert_compatibility is False:
            print(f"Warning: could not locate param {key} in configs")
        return default
    
    @staticmethod
    def to_absolute_path(file_path: Union[Path, str]) -> Path:
        
        """Utility function to convert relative path to absolute path

        Args:
            file_path (Union[Path, str]): Path to file

        Returns:
            Path: Path object with absolute path to file
        """
        return Path(file_path).absolute()


class PipelineConfigs(Configs):
    """Subclass of Configs specific to running pipeline. This is the main entry-point config for running the application `runner.run(pipeline_config)`
    and contains references to other Configs.
    """
    def __init__(self, file_path_str: str) -> None:
        super().__init__(file_path_str)
    
    def assert_compatibility(self):
        """Ability to fail early is configs yaml is not compatible with expected pipeline arguments
        """
        data_configs_path: str = self.get_from_configs('DATA_CONFIGS_PATH', param_type='DATA', 
                                                       is_assert_compatibility=True)
        assert data_configs_path is not None and len(data_configs_path) > 0, (
            "DATA_CONFIGS_PATH not specified in Pipeline Configs")
        
        model_configs_path: List[str] = self.get_from_configs('MODEL_CONFIGS_PATHS', param_type='MODELS', 
                                                       is_assert_compatibility=True)
        assert model_configs_path is not None and len(model_configs_path) > 0, (
            "MODEL_CONFIGS_PATHS not specified in Pipeline Configs")


class DataConfigs(Configs):
    """Suclass of Configs specific to data info. 
    """
    def __init__(self, file_path_str: str) -> None:
        super().__init__(file_path_str)
        
    def assert_compatibility(self):
        """Ability to fail early is configs yaml is not compatible with expected pipeline arguments
        """
        data_type: str = self.get_from_configs('PATH', param_type='DATA', is_assert_compatibility=True)
        assert data_type is not None and len(data_type) > 0, "Data PATH not specified in Data Configs"
        
        data_type: str = self.get_from_configs('TYPE', param_type='DATA', is_assert_compatibility=True)
        assert data_type in ['csv'], "Data TYPE from Data Configs not not supported"
        
        has_numerical_feats = False
        has_grouped_feats = False
        feature_types = ['NUMERICAL', 'CATEGORICAL', 'BINARY', 'GROUPED']
        tot_features_ct = 0
        for f_type in feature_types:
            feats: Optional[List[str]] = self.get_from_configs(f_type, 
                                                               param_type='FEATURES', 
                                                               is_assert_compatibility=True)
            if feats is not None:
                tot_features_ct += len(feats)
                if feats == 'NUMERICAL':
                    has_numerical_feats = True
                if feats == 'GROUPED':
                    has_grouped_feats = True
        assert tot_features_ct > 0, f"No features of types {feature_types} specified in Data Configs"
        
        target: List[str] = self.get_from_configs('TARGET', param_type='FEATURES', is_assert_compatibility=True)
        assert target is not None and len(target) > 0, "TARGET not specified in Data Configs"
        
        numerical_perturbation_intesities: Optional[List[float]] = self.get_from_configs('NUMERICAL_INTENSITIES', 
                                                                                         param_type='PERTURBATIONS',
                                                                                         is_assert_compatibility=True)
        if has_numerical_feats is True:
            assert numerical_perturbation_intesities is not None and len(numerical_perturbation_intesities) > 0, (
                "NUMERICAL_INTENSITIES not specified in Data Configs but NUMERICAL features are specified")
        grouped_perturbation_intesities: Optional[List[float]] = self.get_from_configs('GROUPED_INTENSITIES', 
                                                                                       param_type='PERTURBATIONS',
                                                                                       is_assert_compatibility=True)
        if has_grouped_feats is True:
            assert grouped_perturbation_intesities is not None and len(grouped_perturbation_intesities) > 0, (
                "GROUPED_INTENSITIES not specified in Data Configs but GROUPED features are specified")
            
        output_type: str = self.get_from_configs('OUTPUT_TYPE', param_type='PERTURBATIONS', is_assert_compatibility=True)
        assert output_type in ['logits', 'odds_ratios', 'relative_risk'], (
            "output_type should be one of ['logits', 'odds_ratios', 'relative_risk']")

class ModelConfigs(Configs):
    def __init__(self, file_path_str: str) -> None:
        super().__init__(file_path_str)
        
    def assert_compatibility(self):
        """Ability to fail early is configs yaml is not compatible with expected pipeline arguments
        """
        model_module_name: str = self.get_from_configs('SOURCE_MODULE', 
                                                       param_type='MODEL_INFO', 
                                                       is_assert_compatibility=True)
        assert model_module_name.lower() in ['sklearn', 'keras'], "SOURCE MODULE from Model Configs not supported"
    
    def get_model_args(self) -> Optional[Dict[str, Any]]:
        """Getter method to get model arguments from configs

        Returns:
            Optional[Union[str, Dict[str, Any]]]: Get the model args that are used to instanstiate a model.
        """
        return self.get_from_configs('ARGUMENTS', param_type='MODEL_PARAMS')
    
    def get_build_model_params(self) -> Dict[str, Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """Getter method to get model params from configs

        Returns:
            Optional[Union[str, Dict[str, Any]]]: Model params used to build deep model layers.
        """
        assert self.get_from_configs('SOURCE_MODULE', param_type='MODEL_INFO') == 'keras', (
        "BUILD_MODEL params are only applicable to keras (deep) models")
        
        return self.get_from_configs('BUILD_MODEL', param_type='MODEL_PARAMS')
