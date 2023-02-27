from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC

from src.table15.models.model import Model


class SVM(Model):
    def __init__(self, name: str, model_type: str, model_args: Dict[str, Any]) -> None:
        super().__init__(name, model_type)
        self.model_args: Dict[str, Any] = model_args
        self.feature_importances: Dict[str, float]
    
    @abstractmethod
    def instantiate_model(self) -> SVM:
        pass
    
    def fit(self, X, y) -> Model:
        self.model.fit(X, y.to_numpy().ravel())
        return self
    
        
class SklearnSVM(SVM):
    def __init__(self, name: str, model_type: str, model_args: Dict[str, Any]) -> None:
        super().__init__(name, model_type, model_args)
        self.model: SVC = SVC
        
    def instantiate_model(self) -> SklearnSVM:
        # We require predict_proba
        # See https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
        assert self.model_args["probability"] == True, "Requires `probability: true` in configs"
        self.model = self.model(**self.model_args)
        return self
    
    @abstractmethod
    def extract_feature_importances(self) -> np.array:
        pass
    
    def predict(self, data):
        return self.model.predict_proba(data)[:, 1].ravel()


class SklearnLinearSVM(SklearnSVM):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]]) -> None:
        super().__init__(name, model_type, model_args)
        self.model = CalibratedClassifierCV
        
    def instantiate_model(self) -> SklearnLinearSVM:
        temp_model_args = self.model_args.copy()
        calibrated_classifier_cv_args = temp_model_args.pop("calibrated_classifier_cv_args", {})
        self.model = self.model(LinearSVC(**temp_model_args), **calibrated_classifier_cv_args)
        return self
    
    def extract_feature_importances(self) -> np.array:
        coef_avg = 0
        for i in self.model.calibrated_classifiers_:
            coef_avg = coef_avg + i.base_estimator.coef_
        coef_avg  = coef_avg/len(self.model.calibrated_classifiers_)

        return np.array(coef_avg)
    

class SklearnKernelSVM(SklearnSVM):
    def __init__(self, name: str, model_type: str, model_args: Optional[Dict[str, Any]]) -> None:
        super().__init__(name, model_type, model_args)
        self.kernel: str = self.model_args["kernel"]
    
    def extract_feature_importances(self) -> np.array:
        if self.kernel == "linear":
            return self.model.coef_.ravel()
        raise NotImplementedError
