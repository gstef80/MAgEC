from collections import defaultdict

import numpy as np
import shap
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ModelUtils:
    def __init__(self, x_train, y_train, x_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test


    def model_constructor(self, model_name):
        if model_name == "lr":
            return LogisticRegression(C=1.)
        if model_name == "rf":
            return CalibratedClassifierCV(RandomForestClassifier(n_estimators=1000), cv=5, method='sigmoid')
        if model_name == "svm":
            return SVC(probability=True)
        if model_name == "mlp":
            return self.create_mlp()
        
        raise ValueError(f"Model {model_name} not found!")


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
        
    def mean_calibrated_clf_cv_importances(self, calibrated_clf_cv):
        calibrated_clfs = calibrated_clf_cv.calibrated_classifiers_
        features = calibrated_clf_cv.feature_names_in_
        
        feat_imps = np.zeros(shape=(len(calibrated_clfs), len(features)))
        for i, calibrated_clf in enumerate(calibrated_clfs):
            feat_imps[i] = calibrated_clf.base_estimator.feature_importances_
        return np.mean(feat_imps, axis=0)
    
    def get_fit_params(self, model_name):
        if model_name == "mlp":
            return {"epochs": 100, "batch_size": 64, "verbose": 0}
        return {}
        
        
    def train_single_model(self, model_name):
        clf = self.model_constructor(model_name)
        fit_params = self.get_fit_params(model_name)
        clf.fit(self.x_train, self.y_train, **fit_params)
        return clf
    
    
    def train_models(self, models):
        models_dict = dict()
        for model_name in models:
            clf = self.train_single_model(model_name)
            models_dict[model_name] = clf
            print(f"Finished training {model_name} model")
        # # Seems to be an issue using KerasClassifier (for ensemble) with a pretrained model when calling predict downstream
        # if use_ensemble:
        #     # create our voting classifier, inputting our models
        #     ensemble = VotingClassifier(estimators, voting='soft')
        #     ensemble._estimator_type = "classifier"
        #     ensemble.fit(x_train_p, y_train_p)
        return models_dict

    
    def l2_normalize_list(self, li, is_abs=False):
        li = np.array(li)
        norm = np.linalg.norm(li)
        ret = li / norm
        if is_abs:
            ret = np.abs(ret)
        return ret


    def extract_feature_importance_from_models(self, models_dict):
        model_feat_imp_dict = defaultdict(dict)
        features = self.x_train.columns
        for model_name, clf in models_dict.items():
            model_feature_importance = self.extract_model_feature_importance(model_name, clf)
            # Normalize feature importance per model
            model_feature_importance = self.l2_normalize_list(model_feature_importance, is_abs=True)
            feature_to_importance = dict(zip(features, model_feature_importance))
            model_feat_imp_dict[model_name] = feature_to_importance 
        return model_feat_imp_dict


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
        background = shap.sample(self.x_train.to_numpy(), 50)
        if explainer_type == 'kernel':
            explainer = shap.KernelExplainer(model.predict, background)
        elif explainer_type == 'deep':
            explainer = shap.DeepExplainer(model, background)
            
        shap_values = explainer.shap_values(shap.sample(self.x_test.to_numpy(), 50))
        shap_means = np.mean(shap_values, axis=1)
        l2_norm = np.linalg.norm(shap_means)
        normalized_shap_means = np.abs(shap_means) / l2_norm
        return normalized_shap_means.ravel()
