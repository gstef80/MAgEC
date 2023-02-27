from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from keras.layers import Dense, Dropout
from keras.models import Sequential

from src.table15.configs import ModelConfigs
from src.table15.models.model import Model
from src.table15.models.model_factory import ModelFactory


class ModelsContainer:
    def __init__(self):
        self.models: List[Model] = []
        self.models_dict: Dict[str, BaseModel] = {}
        self.model_feat_imp_dict: Dict[str, Dict[str, float]] = defaultdict(dict)

    def load_models(self, model_configs_paths: List[str], ensemble_configs_path: Optional[str]=None) -> ModelsContainer:
        for config_path in model_configs_paths:
            m_config = ModelConfigs(config_path)
            model = self.construct_model(m_config) \
                .instantiate_model()
            self.models.append(model)
        if ensemble_configs_path is not None:
            m_config = ModelConfigs(ensemble_configs_path)
            estimators = [(m.name, m.model) for m in self.models]
            model = self.construct_model(m_config) \
                .set_estimators(estimators) # type: ignore
            model = model.instantiate_ensemble_model() 
            self.models.append(model)
        return self

    def populate_data_tables(self, x_train, Y_train, x_test=None) -> ModelsContainer:
        self.x_train = x_train
        self.Y_train = Y_train
        self.x_test = x_test
        return self

    def construct_model(self, model_configs: ModelConfigs) -> Model:
        return ModelFactory.construct_model(model_configs)
    
    def train_models(self) -> ModelsContainer:
        if len(self.models) == 0:
            raise ValueError("No models generated to train...")
        print('Training models ...')
        for model in self.models:
            model = model.fit(self.x_train, self.Y_train)
            self.models_dict[model.name] = model
            print(f"Finished training {model.name} model")
        print(f'Finished generating models {list(self.models_dict.keys())}')
        # TODO: add ensemble here
        # Seems to be an issue using KerasClassifier (for ensemble) with a 
        # pretrained model when calling predict downstream
        return self
    
    def store_feature_importance_from_models(self, use_feature_importance_scaling: bool=True) -> ModelsContainer:
        if len(self.models_dict) == 0:
            raise ValueError("Models have not been trained yet")
        if use_feature_importance_scaling is True:
            features = self.x_train.columns
            for model_name, model in self.models_dict.items():
                model_feature_importances = model.extract_feature_importances()
                model_feature_importances = self.l2_normalize_list(model_feature_importances, is_abs=True)
                feature_to_importance = dict(zip(features, model_feature_importances))
                self.model_feat_imp_dict[model_name] = feature_to_importance 
            print("Extracted model feature importances")
        return self
    
    def l2_normalize_list(self, li: np.array, is_abs: bool=False) -> np.array:
        norm = np.linalg.norm(np.array(li))
        return np.abs(li / norm) if is_abs is True else li / norm
    
    def extract_model_feature_importance(self, model_name, clf):
        if model_name == "lr":
            return clf.coef_.ravel()
        if model_name == "rf":
            return self.mean_calibrated_clf_cv_importances(clf)
        if model_name == "svm":
            return self.get_shap_values(clf, explainer_type="kernel")
        if model_name == "mlp":
            return self.get_shap_values(clf, explainer_type="deep")
        
        raise ValueError(f"Model {model_name} not found!")
    
    def get_fit_params(self, model_name):
        if model_name == "mlp":
            return {"epochs": 100, "batch_size": 64, "verbose": 0}
        return {}
    
    def create_mlp(self):
        mlp = Sequential()
        mlp.add(Dense(60, input_dim=len(self.x_train.columns), activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(30, input_dim=60, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(1, activation='sigmoid'))
        mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return mlp

    def get_shap_values(self, model, explainer_type):
        if explainer_type == 'kernel':
            def model_predict(data_asarray):
                data_asframe = pd.DataFrame(data_asarray, columns=model_features)
                return model.predict(data_asframe)
            NUM_CENTROIDS = 10
            model_features = self.x_train.columns
            x_train_kmeans = shap.kmeans(self.x_train, NUM_CENTROIDS)
            explainer = shap.KernelExplainer(model_predict, np.array(x_train_kmeans.data))
        elif explainer_type == 'deep':
            background = shap.sample(self.x_train.to_numpy(), 50)
            explainer = shap.DeepExplainer(model, background)
            
        shap_values = explainer.shap_values(shap.sample(self.x_test.to_numpy(), 50))
        shap_means = np.mean(shap_values, axis=0)
        l2_norm = np.linalg.norm(shap_means)
        normalized_shap_means = np.abs(shap_means) / l2_norm
        return normalized_shap_means.ravel()
