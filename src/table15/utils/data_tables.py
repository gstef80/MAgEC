from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.table15.configs import Configs, DataConfigs


class DataTables:
    def __init__(self):
        self.data_configs: DataConfigs = None
        self.numerical_features = None
        self.binary_features = None
        self.categorical_features = None
        self.grouped_features = None
        self.x_train = None
        self.x_test = None
        self.Y_train = None
        self.Y_test = None
        self.test_stats_dict = {}
        self.setted_numerical_values = None
    
    def set_data_configs(self, data_configs_file_path: str) -> DataTables:
        self.data_configs = DataConfigs(data_configs_file_path)
        return self
    
    def generate_data(self) -> DataTables:
        assert self.data_configs is not None, (
            "DataConfigs has not been set in DataTables object")
        
        def impute(df):
            out = df.copy()
            cols = df.columns
            out[cols] = out[cols].replace(0, np.NaN)
            out[cols] = out[cols].fillna(out[cols].mean())
            return out

        csv_path = self.data_configs.get_from_configs("PATH", param_type="DATA")
        csv_path = self.data_configs.to_absolute_path(csv_path)

        numerical_features = self.data_configs.get_from_configs("NUMERICAL", param_type="FEATURES")
        categorical_features = self.data_configs.get_from_configs("CATEGORICAL", param_type="FEATURES")
        binary_features = self.data_configs.get_from_configs("BINARY", param_type="FEATURES")
        target_feature = self.data_configs.get_from_configs("TARGET", param_type="FEATURES")
        self.grouped_features = self.data_configs.get_from_configs("GROUPED", param_type="FEATURES")
        
        self.setted_numerical_values = self.data_configs.get_from_configs("SETTED_NUMERICAL_VALUES", 
                                                                          param_type="FEATURES",
                                                                          default={})
        n_data_samples = self.data_configs.get_from_configs("N_DATA_SAMPLES", param_type="DATA")
        test_size = self.data_configs.get_from_configs("TEST_SIZE", param_type="DATA")
        random_seed = self.data_configs.get_from_configs("RANDOM_SEED", param_type="DATA")

        if random_seed is not None:
            np.random.seed(random_seed)
        
        df = pd.read_csv(csv_path)
        if n_data_samples is not None and n_data_samples > 0:
            df = df.sample(n=n_data_samples)

        x_num = df.loc[:, numerical_features]
        x_num = impute(x_num)
        self.numerical_features = numerical_features

        x_bin = df.loc[:, binary_features]
        self.binary_features = binary_features

        x_cat = df.loc[:, categorical_features].fillna("")
        if not x_cat.empty:
            # use prefix_sep to delimitate later
            x_cat = pd.get_dummies(x_cat, prefix_sep="__cat__")
        self.categorical_features = list(x_cat.columns)
        
        non_numerical_features = self.binary_features + self.categorical_features

        x = pd.concat([x_num, x_bin, x_cat], axis=1)

        Y = df.loc[:, target_feature]

        x_train, self.x_test, Y_train, self.Y_test = train_test_split(x, Y, test_size=test_size, random_state=random_seed)
        
        # Only test (get Magecs for) sick patients
        # self.Y_test = self.Y_test[self.Y_test[target_feature[0]] == 1.]
        # self.x_test = self.x_test[self.x_test.index.isin(self.Y_test.index)]

        stsc = StandardScaler()
        
        xst_train = x_train.copy()
        xst_test = self.x_test
        if len(numerical_features) > 0:
            xst_train = stsc.fit_transform(x_train[numerical_features])
            xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=numerical_features)
            xst_train = pd.concat([xst_train, x_train[non_numerical_features]], axis=1)

            xst_test = stsc.transform(self.x_test[numerical_features])
            xst_test = pd.DataFrame(xst_test, index=self.x_test.index, columns=numerical_features)
            xst_test = pd.concat([xst_test, self.x_test[non_numerical_features]], axis=1)
            
            self.rescale_set_feature_values_dict(stsc, numerical_features)

        # Format
        self.x_train = self.format_df(xst_train.copy())
        self.Y_train = self.format_df(Y_train.copy())
        self.x_test = self.format_df(xst_test.copy())
        self.Y_test = self.format_df(self.Y_test.copy())
        
        
        self.generate_test_stats()

        return self
    
    
    def format_df(self, df):
        df = pd.DataFrame(df)
        df["timepoint"] = 0
        df["case"] = np.arange(len(df))
        df.set_index(["case", "timepoint"], inplace=True)
        df.sort_index(axis=1, inplace=True)
        return df
    
    
    def rescale_set_feature_values_dict(self, stsc, numerical_features):
        scaling_df = pd.DataFrame([[None for _ in range(len(numerical_features))]], columns=numerical_features)
        for feat, val in self.setted_numerical_values.items():
            scaling_df[feat] = val
        
        scaled_values = dict(zip(numerical_features, stsc.transform(scaling_df[numerical_features]).ravel()))
        for feat, val in self.setted_numerical_values.items():
            if feat in numerical_features:
                self.setted_numerical_values[feat] = scaled_values[feat]

    def generate_test_stats(self):
        means = self.x_test.mean()
        self.test_stats_dict["mean"] = means
        
        stds = self.x_test.std()
        self.test_stats_dict["std"] = stds
        
        meadians = self.x_test.median()
        self.test_stats_dict["median"] = meadians

    def get_features_by_type(self, feature_type):
        features_type_to_features = {
            "numerical": self.numerical_features,
            "binary": self.binary_features,
            "categorical": self.categorical_features,
            "grouped": self.grouped_features
        }
        return features_type_to_features.get(feature_type, None)
        
