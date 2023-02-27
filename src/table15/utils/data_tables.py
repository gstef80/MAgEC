from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.table15.configs import Configs, DataConfigs


class DataTables:
    def __init__(self):
        self.data_configs: DataConfigs
        self.numerical_features: List[str]
        self.binary_features: List[str]
        self.categorical_features: List[str]
        self.grouped_features: List[str]
        self.x_train: pd.DataFrame
        self.x_test: pd.DataFrame
        self.Y_train: pd.DataFrame
        self.Y_test: pd.DataFrame
        self.test_stats_dict: Dict[str, Dict[str, pd.Series]]
        self.setted_numerical_values: Dict[str, float]
    
    def set_data_configs(self, data_configs_file_path: str) -> DataTables:
        """Setter method to set configs arugments to DataTables

        Args:
            data_configs_file_path (str): string filepath to data configs Yaml

        Returns:
            DataTables: self
        """
        self.data_configs = DataConfigs(data_configs_file_path)
        return self
    
    def generate_data(self) -> DataTables:
        """Main method to generate and store important data tables and data stats to class membership.
        Steps include:
        1) Configs and data parameters setup
        2) Read data and sample it if applicable
        3) Differentiate data columns by feature type (eg: numerical, binary, categorical)
        4) Impute on numerical features
        5) Set x, Y, and perform train-test-split
        6) Run Standard Scaler on numerical features only for train and test sets
        7) Generate test stats from non-scaled feature values to display on output table
        8) Format dataframes to include helper columns, and store dataframes in class membership

        Returns:
            DataTables: self
        """
        assert self.data_configs is not None, "DataConfigs has not been set in DataTables object"
        
        def impute(df):
            out = df.copy()
            cols = df.columns
            out[cols] = out[cols].replace(0, np.NaN)
            out[cols] = out[cols].fillna(out[cols].mean())
            return out

        csv_path = self.data_configs.get_from_configs('PATH', param_type='DATA')
        csv_path = self.data_configs.to_absolute_path(csv_path)

        numerical_features = self.data_configs.get_from_configs('NUMERICAL', param_type='FEATURES')
        categorical_features = self.data_configs.get_from_configs('CATEGORICAL', param_type='FEATURES')
        binary_features = self.data_configs.get_from_configs('BINARY', param_type='FEATURES')
        target_feature = self.data_configs.get_from_configs('TARGET', param_type='FEATURES')
        self.grouped_features = self.data_configs.get_from_configs('GROUPED', param_type='FEATURES')
        
        self.setted_numerical_values = self.data_configs.get_from_configs('SETTED_NUMERICAL_VALUES', 
                                                                          param_type='FEATURES',
                                                                          default={})
        n_data_samples = self.data_configs.get_from_configs('N_DATA_SAMPLES', param_type='DATA')
        test_size = self.data_configs.get_from_configs('TEST_SIZE', param_type='DATA')
        random_seed = self.data_configs.get_from_configs('RANDOM_SEED', param_type='DATA')
        only_test_positive_class =  self.data_configs.get_from_configs('ONLY_TEST_POSITIVE_CLASS', param_type='DATA')

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

        x_cat = df.loc[:, categorical_features].fillna('')
        if not x_cat.empty:
            # use prefix_sep to delimitate later
            x_cat = pd.get_dummies(x_cat, prefix_sep='__cat__')
        self.categorical_features = list(x_cat.columns)
        
        non_numerical_features = self.binary_features + self.categorical_features

        x = pd.concat([x_num, x_bin, x_cat], axis=1)

        Y = df.loc[:, target_feature]

        x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=test_size, random_state=random_seed)
        
        if only_test_positive_class == True:
            # Only test (get Magecs for) positive class cases.
            Y_test = Y_test[Y_test[target_feature[0]] == 1.]
            x_test = x_test[x_test.index.isin(Y_test.index)]

        stsc = StandardScaler()
        
        xst_train = x_train[numerical_features].copy()
        xst_test = x_test[numerical_features].copy()
        if len(numerical_features) > 0:
            xst_train = stsc.fit_transform(xst_train)
            xst_train = pd.DataFrame(xst_train, index=x_train.index, columns=numerical_features) # type: ignore
            xst_train = pd.concat([xst_train, x_train[non_numerical_features]], axis=1)

            xst_test = stsc.transform(xst_test)
            xst_test = pd.DataFrame(xst_test, index=x_test.index, columns=numerical_features) # type: ignore
            xst_test = pd.concat([xst_test, x_test[non_numerical_features]], axis=1)
            
            self.rescale_set_feature_values_dict(stsc, numerical_features)
        
        self.generate_test_stats(x_test) # type: ignore
        # Format
        self.x_train = self.format_df(xst_train)
        self.Y_train = self.format_df(pd.DataFrame(Y_train))
        self.x_test = self.format_df(xst_test)
        self.Y_test = self.format_df(pd.DataFrame(Y_test))

        return self
    
    def format_df(self, df: pd.DataFrame) ->  pd.DataFrame:
        """Add helper columns to dataframe to ID individuals and create timepoints for future time-series model implementations.

        Args:
            df (pd.DataFrame): Pandas dataframe containing features of a single type (numerical, binary, etc..).

        Returns:
            pd.DataFrame: Pandas dataframe with helper columns added as index and sorted.
        """
        df = pd.DataFrame(df)
        df['timepoint'] = 0
        df['case'] = np.arange(len(df))
        df.set_index(['case', 'timepoint'], inplace=True)
        df.sort_index(axis=1, inplace=True)
        return df
    
    def rescale_set_feature_values_dict(self, stsc: StandardScaler, numerical_features: List[str]) -> None:
        """When using set feature values as real-world values, this function rescales these using the same Standard Scaler
        to perform correct perturbations to these set values.

        Args:
            stsc (StandardScaler): The scaler used to scale numerical features.
            numerical_features (List[str]): a list of the numerical features obtained from Data Configs
        """
        scaling_df = pd.DataFrame([[None for _ in range(len(numerical_features))]], columns=numerical_features)
        for feat, val in self.setted_numerical_values.items():
            scaling_df[feat] = val
        
        scaled_values = dict(zip(numerical_features, stsc.transform(scaling_df[numerical_features]).ravel())) # type: ignore
        for feat, val in self.setted_numerical_values.items():
            if feat in numerical_features:
                self.setted_numerical_values[feat] = scaled_values[feat]

    def generate_test_stats(self, x_test: pd.DataFrame) -> None:
        """Calculates useful statistics for x_test before StandardScaler is applied to it. These stats are displayed in the 
        final output.
        """
        self.test_stats_dict = {}
        self.test_stats_dict['numerical'] = {}
        x_test_num = x_test[self.numerical_features]
        self.test_stats_dict['numerical']['mean'] = x_test_num.mean()
        self.test_stats_dict['numerical']['std'] = x_test_num.std()
        self.test_stats_dict['numerical']['median'] = x_test_num.median()
        
        self.test_stats_dict['binary'] = {}
        x_test_bin = x_test[self.binary_features]
        self.test_stats_dict['binary']['prevalence'] = x_test_bin.mean()
        
        self.test_stats_dict['categorical'] = {}
        x_test_cat = x_test[self.categorical_features]
        self.test_stats_dict['categorical']['counts'] = x_test_cat.sum()

    def get_features_by_type(self, feature_type: str) -> List[str]:
        """Convert string of feature type to 

        Args:
            feature_type (_type_): _description_

        Returns:
            List[str]: _description_
        """
        if feature_type not in ["numerical", "binary", "categorical", "grouped"]:
            raise ValueError('Feature type must be numerical, binary, categorical, or grouped')
        
        features_type_to_features = {
            'numerical': self.numerical_features,
            'binary': self.binary_features,
            'categorical': self.categorical_features,
            'grouped': self.grouped_features
        }
        return features_type_to_features[feature_type]
