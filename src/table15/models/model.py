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
    def extract_feature_importances(self):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class Model(BaseModel):
    def __init__(self, name: str, model_type: str) -> None:
        super().__init__()
        self.name: str = name
        self.model_type: str = model_type
        self.model: Model

    @abstractmethod
    def instantiate_model(self) -> Model:
        pass