from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from keras.layers import Dense, Dropout, Layer
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from src.table15.models.model import Model


class DeepModels(Model):
    def __init__(self, 
                 name: str, 
                 model_type: str, 
                 model_args: Dict[str, Any], 
                 build_model_params: Dict[str, Union[List[Dict[str, Any]], Dict[str, Any]]]) -> None:
        super().__init__(name, model_type)
        self.model_args: Dict[str, Any] = model_args
        self.build_model_params: Dict[str, Union[List[Dict[str, Any]], Dict[str, Any]]] = build_model_params
        self.feature_importances: Dict[str, float]
        
    @abstractmethod
    def instantiate_model(self) -> DeepModels:
        pass
    
    @abstractmethod
    def build_model(self) -> DeepModels:
        pass
    
    def fit(self, X, y) -> Model:
        self.model.fit(X, y.to_numpy().ravel())
        return self
    
    def predict(self, data) -> np.array:
        return np.array(self.model.predict_proba(data))[:, 1].ravel()
        
        
class KerasMultiLayerPerceptron(DeepModels):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]], 
                 build_model_params: Dict[str, Union[List[Dict[str, Any]], Dict[str, Any]]]) -> None:
        super().__init__(name, model_type, model_args, build_model_params)
        self.model: KerasClassifier = KerasClassifier
        self.str_to_layer: Dict[str, Layer] = {
            'Dense': Dense, 'Dropout': Dropout}
        
    def instantiate_model(self) -> DeepModels:
        self.model = self.model(build_fn=self.build_model, **self.model_args)
        self.model._estimator_type = "classifier"
        return self
        
    def build_model(self) -> Sequential:
        adders = self.build_model_params['add']
        compiler_params = self.build_model_params['compile']
        mlp = Sequential()
        for layer_params in adders:
            layer_params_copy = layer_params.copy()
            layer_obj = self.str_to_layer[layer_params_copy['type']]
            _ = layer_params_copy.pop('type')
            mlp.add(layer_obj(**layer_params_copy))
        mlp.compile(**compiler_params)
        return mlp
    
    def extract_feature_importances(self) -> np.array:
        raise NotImplementedError
