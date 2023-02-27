from src.table15.configs import ModelConfigs
from src.table15.models.deep_models import KerasMultiLayerPerceptron
from src.table15.models.ensemble_models import SklearnRandomForest, SklearnVotingClassifier
from src.table15.models.linear_models import (SklearnLogisticRegression,
                                              SklearnLogisticRegressionCV)
from src.table15.models.model import Model
from src.table15.models.svm_models import SklearnKernelSVM, SklearnLinearSVM


class ModelFactory:
    @staticmethod
    def construct_model(model_configs: ModelConfigs) -> Model:
        """Factory method to build any model, given that it has been implemented.

        Args:
            model_configs (ModelConfigs): Configs object generated from model configs Yaml.

        Raises:
            ValueError: If the model is not found based on the description in model_configs, raise an error.

        Returns:
            Model: A model object that contains one of many implemented model types.
        """
        model_name = model_configs.get_from_configs('NAME', param_type='MODEL_INFO')
        model_type = model_configs.get_from_configs('TYPE', param_type='MODEL_INFO').lower()
        source_module = model_configs.get_from_configs('SOURCE_MODULE', param_type='MODEL_INFO').lower()
        model_args = model_configs.get_model_args()
        
        if source_module == 'sklearn':
            if model_type == 'logistic_regression':
                return SklearnLogisticRegression(model_name, model_type, model_args)
            if model_type == 'logistic_regression_cv':
                return SklearnLogisticRegressionCV(model_name, model_type, model_args)
            if model_type == 'linear_svm':
                return SklearnLinearSVM(model_name, model_type, model_args)
            if model_type == 'kernel_svm':
                return SklearnKernelSVM(model_name, model_type, model_args)
            if model_type == 'random_forest':
                return SklearnRandomForest(model_name, model_type, model_args)
            if model_type == 'voting_classifier':
                return SklearnVotingClassifier(model_name, model_type, model_args)
        if source_module == 'keras':
            build_model_params = model_configs.get_build_model_params()
            if model_type == 'multi_layer_perceptron':
                return KerasMultiLayerPerceptron(model_name, model_type, model_args, build_model_params)
        
        raise ValueError(f"Model {model_name} not found!")
