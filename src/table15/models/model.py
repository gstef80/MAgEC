from __future__ import annotations

from abc import ABC, abstractmethod
from ast import Dict
from typing import Any

import numpy as np
from scikeras.wrappers import KerasClassifier



class BaseModel(ABC):
    
    @abstractmethod
    def fit(self, X, y) -> BaseModel:
        pass
    
    @abstractmethod
    def extract_feature_importances(self) -> np.array:
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class Model(BaseModel):
    def __init__(self, name: str, model_type: str) -> None:
        super().__init__()
        self.name: str = name
        self.model_type: str = model_type
    
    def fit(self, X, y) -> Model:
        return super().fit(X, y)
    
    def extract_feature_importances(self) -> np.array:
        return super().extract_feature_importances()
    
    def predict(self, X):
        return super().predict(X)

            