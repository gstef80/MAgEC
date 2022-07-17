from typing import Dict
import yaml

def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        parsed_yaml = yaml.safe_load(file)
    return parsed_yaml


def get_from_configs(configs: Dict, key: str, param_type: str=None):
    if param_type == 'hyperparams':
        for k in configs:
                if k == key:
                    return configs[k]
    for k in configs:
        if k == key:
            return configs[k]
    return None
