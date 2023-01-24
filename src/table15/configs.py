from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Configs:
    def __init__(self, file_path_str: str) -> None:
        self.configs_dict: Dict[str, Any]
        self.file_path: Path
        self.configs_dict = self.import_configs(file_path_str)
        
    def import_configs(self, file_path_str: str) -> Configs:
        self.file_path = Path(file_path_str)
        with open(self.file_path, 'r') as file:
            return yaml.safe_load(file)
        
    def get_from_configs(self, key: str, param_type: str="CONFIGS") -> Optional[Dict[str, Any]]:
        key = key.upper()
        if param_type in self.configs_dict and key in self.configs_dict[param_type]:
            return self.configs_dict[param_type][key]
        if key in self.configs_dict:
            return self.configs_dict[key]
        print(f"Warning: could not locate param {key} in configs")


class EmptyConfigs(Configs):
    def __init__(self, file_path_str: str=None) -> None:
        super().__init__(file_path_str)


class ModelConfigs(Configs):
    def __init__(self, file_path_str: str) -> None:
        super().__init__(file_path_str)
        
        self.assert_model_compatibiliy()
        
    def assert_model_compatibiliy(self):
        model_module_name = str(self.get_from_configs("SOURCE_MODULE", param_type="MODEL_INFO"))
        assert model_module_name.lower() in ["sklearn", "keras"]
    
    def get_model_args(self):
        return self.get_from_configs("ARGUMENTS", param_type="MODEL_PARAMS")

