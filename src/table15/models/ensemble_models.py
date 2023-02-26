from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from src.table15.models.model import Model


class EnsembleModels(Model):
    def __init__(self, name: str, model_type: str, model_args: Dict[str, Any]) -> None:
        super().__init__(name, model_type)
        self.model_args: Dict[str, Any] = model_args
        self.feature_importances: Dict[str, float]
        
    @abstractmethod
    def instantiate_model(self) -> EnsembleModels:
        pass
    
    def fit(self, X, y) -> Model:
        self.model.fit(X, y.to_numpy().ravel())
        return self

    def predict(self, data):
        return self.model.predict_proba(data)[:, 1].ravel()
    

class SklearnRandomForest(EnsembleModels):
    def __init__(self, name: str, model_type: str, model_args: Dict[str, Any]) -> None:
        super().__init__(name, model_type, model_args)
        self.model: Union[RandomForestClassifier, CalibratedClassifierCV] = RandomForestClassifier
        self.is_calibrated_classifier: bool = False
    
    def instantiate_model(self) -> SklearnRandomForest:
        temp_model_args = self.model_args.copy()
        calibrated_classifier_cv_args = temp_model_args.pop("calibrated_classifier_cv_args", {})
        if len(calibrated_classifier_cv_args) > 0:
            self.is_calibrated_classifier = True
            self.model = CalibratedClassifierCV(self.model(**temp_model_args), **calibrated_classifier_cv_args)
        else:
            self.model = self.model(**self.model_args)
        return self
    
    def extract_feature_importances(self) -> np.array:
        if self.is_calibrated_classifier is True:
            calibrated_clfs = self.model.calibrated_classifiers_
            features = self.model.feature_names_in_

            feat_imps = np.zeros(shape=(len(calibrated_clfs), len(features)))
            for i, calibrated_clf in enumerate(calibrated_clfs):
                feat_imps[i] = calibrated_clf.base_estimator.feature_importances_
            all_importances = np.mean(feat_imps, axis=0, dtype=np.float64)
            return all_importances / np.sum(all_importances)
        
        else:
            return self.model.feature_importances_


class SklearnVotingClassifier(EnsembleModels):
    def __init__(self, name: str, model_type: str, model_args: Dict[str, Any]) -> None:
        super().__init__(name, model_type, model_args)
        self.estimators: List[Model]
        self.model: VotingClassifier = VotingClassifier
            
    def instantiate_model(self) -> SklearnVotingClassifier:
        raise NotImplementedError
    
    def set_estimators(self, estimators: List[Model]) -> SklearnVotingClassifier:
        self.estimators = estimators
        return self
    
    def instantiate_ensemble_model(self) -> SklearnVotingClassifier:
        assert len(self.estimators) > 1, "Voting Classifier does not have enough estimators"
        self.model = self.model(self.estimators, **self.model_args)
        return self
    
    def extract_feature_importances(self) -> np.array:
        raise NotImplementedError
