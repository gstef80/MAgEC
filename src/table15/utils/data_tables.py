from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.table15.configs import Configs


class DataTables:
    def __init__(self):
        self.df = None
        self.features = None
        self.numerical_features = None
        self.binary_features = None
        self.categorical_features = None
        self.grouped_features = None
        self.x_train_p = None
        self.x_validation_p = None
        self.y_train_p = None
        self.y_validation_p = None
        self.x_validation = None
        self.Y_validation = None
        self.validation_stats_dict = {}
        self.set_feature_values = None
    
    def generate_data(self, configs: Configs) -> DataTables:
        def impute(df):
            out = df.copy()
            cols = df.columns
            out[cols] = out[cols].replace(0, np.NaN)
            out[cols] = out[cols].fillna(out[cols].mean())
            return out

        csv_path = configs.get_from_configs('CSV_PATH')

        numerical_features = configs.get_from_configs('NUMERICAL', param_type='FEATURES')
        categorical_features = configs.get_from_configs('CATEGORICAL', param_type='FEATURES')
        binary_features = configs.get_from_configs('BINARY', param_type='FEATURES')
        target_feature = configs.get_from_configs('TARGET', param_type='FEATURES')
        
        self.grouped_features = configs.get_from_configs('GROUPED', param_type='FEATURES')
        self.set_feature_values = configs.get_from_configs('SET_FEATURE_VALUES', param_type='FEATURES')

        random_seed = configs.get_from_configs('RANDOM_SEED', param_type='HYPERPARAMS')
        n_samples = configs.get_from_configs('N_SAMPLES', param_type='CONFIGS')
        test_size = configs.get_from_configs('TEST_SIZE', param_type='HYPERPARAMS')

        if self.set_feature_values is None:
            self.set_feature_values = dict()
            
        self.df = pd.read_csv(csv_path)

        if random_seed is not None:
            np.random.seed(random_seed)
            
        if n_samples is not None:
            self.df = self.df.sample(n=n_samples)

        x_num = self.df.loc[:, numerical_features]
        x_num = impute(x_num)
        self.numerical_features = numerical_features

        x_bin = self.df.loc[:, binary_features]
        self.binary_features = binary_features

        x_cat = self.df.loc[:, categorical_features].fillna('')
        if not x_cat.empty:
            # use prefix_sep to deliminate later
            x_cat = pd.get_dummies(x_cat, prefix_sep="__cat__")
        self.categorical_features = list(x_cat.columns)
        
        non_numerical_features = self.binary_features + self.categorical_features
        self.features = self.numerical_features + non_numerical_features
        # for group in self.grouped_features:
        #     for feat in group:
        #         if feat not in self.features:
        #             self.features += feat

        x = pd.concat([x_num, x_bin, x_cat], axis=1)

        Y = self.df.loc[:, target_feature]

        x_train, self.x_validation, Y_train, self.Y_validation = train_test_split(x, Y, test_size=test_size, random_state=random_seed)
        
        # Only test (get Magecs for) sick patients
        # self.Y_validation = self.Y_validation[self.Y_validation[target_feature[0]] == 1.]
        # self.x_validation = self.x_validation[self.x_validation.index.isin(self.Y_validation.index)]

        stsc = StandardScaler()
        
        xst_train = x_train
        xst_validation = self.x_validation
        if len(numerical_features) > 0:
            xst_train = stsc.fit_transform(x_train[numerical_features])
            xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=numerical_features)
            xst_train = pd.concat([xst_train, x_train[non_numerical_features]], axis=1)

            xst_validation = stsc.transform(self.x_validation[numerical_features])
            xst_validation = pd.DataFrame(xst_validation, index=self.x_validation.index, columns=numerical_features)
            xst_validation = pd.concat([xst_validation, self.x_validation[non_numerical_features]], axis=1)
            
            self.rescale_set_feature_values_dict(stsc, numerical_features)

        # Format
        self.x_train_p = self.format_df(xst_train.copy())
        self.y_train_p = self.format_df(Y_train.copy())
        self.x_validation_p = self.format_df(xst_validation.copy())
        self.y_validation_p = self.format_df(self.Y_validation.copy())
        
        
        self.generate_validation_stats()

        return self
    
    
    def format_df(self, df):
        df = pd.DataFrame(df)
        df['timepoint'] = 0
        df['case'] = np.arange(len(df))
        df.set_index(['case', 'timepoint'], inplace=True)
        df.sort_index(axis=1, inplace=True)
        return df
    
    
    def rescale_set_feature_values_dict(self, stsc, numerical_features):
        scaling_df = pd.DataFrame([[None for _ in range(len(numerical_features))]], columns=numerical_features)
        for feat, val in self.set_feature_values.items():
            scaling_df[feat] = val
        
        scaled_values = dict(zip(numerical_features, stsc.transform(scaling_df[numerical_features]).ravel()))
        for feat, val in self.set_feature_values.items():
            if feat in numerical_features:
                self.set_feature_values[feat] = scaled_values[feat]


    def generate_validation_stats(self):
        means = self.x_validation.mean()
        self.validation_stats_dict["mean"] = means
        
        stds = self.x_validation.std()
        self.validation_stats_dict["std"] = stds
        
        meadians = self.x_validation.median()
        self.validation_stats_dict["median"] = meadians

    def get_features_by_type(self, feature_type):
        features_type_to_features = {
            "numerical": self.numerical_features,
            "binary": self.binary_features,
            "categorical": self.categorical_features,
            "grouped": self.grouped_features
        }
        return features_type_to_features.get(feature_type, None)
        
