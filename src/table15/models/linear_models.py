from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from src.table15.configs import ModelConfigs
from src.table15.models.model import Model


class LinearModel(Model):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]]) -> None:
        super().__init__(name, model_type)
        self.model_args: Dict[str, Any] = model_args
        self.feature_importances: Dict[str, float]
    
    def fit(self, X, y) -> Model:
        self.model.fit(X, y.to_numpy().ravel())
        return self
    
    def extract_feature_importances(self) -> np.array:
        return super().extract_feature_importances()
    
    def predict(self, X):
        return super().predict(X)


class SklearnLogisticRegression(LinearModel):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]]) -> None:
        super().__init__(name, model_type, model_args)
        self.model: LogisticRegression = LogisticRegression
    
    def instantiate_model(self) -> SklearnLogisticRegression:
        self.model = self.model(**self.model_args)
        return self
        
    def extract_feature_importances(self) -> np.array:
        return self.model.coef_.ravel()
    
    def predict(self, data):
        return self.model.predict_proba(data)[:, 1].ravel()
    

class SklearnLogisticRegressionCV(SklearnLogisticRegression):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]]) -> None:
        super().__init__(name, model_type, model_args)
        self.model = LogisticRegressionCV
        
    def instantiate_model(self) -> SklearnLogisticRegressionCV:
        self.model = self.model(**self.model_args)
        return self
        
    def extract_feature_importances(self) -> np.array:
        return super().extract_feature_importances()

    def predict(self, data):
        return super().predict(data)
