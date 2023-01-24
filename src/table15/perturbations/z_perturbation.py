from typing import List
from src.table15.perturbations.perturbation import Perturbation
import numpy as np
import pandas as pd

class ZPerturbation(Perturbation):
    def __init__(self, target_data: pd.DataFrame, model_name: str, features: List[str], feature_type: str) -> None:
        super().__init__(target_data, model_name, features, feature_type)
        
    def run_perturbation(self, set_feature_values, baseline=1.0):
        '''
        Main method for computing a MAgEC. Assumes 'scaled/normalized' features in target data.
            Supporting 2 types of variables:
            - numeric / floats
            - binary / boolean
            Default score_comparison subtracts perturbed output from original.
            For a binary classification task, where 1 denotes a "bad" outcome, a good perturbation
            is expected to result in a negative score_comparison (assuming monotonic score_preprocessing).
        '''
        prob_deltas_per_cell = pd.DataFrame(index=self.target_data.index, columns=self.target_data.columns)
        
        # Predict for original data
        base_df = self.model_predict_probs_and_logits(self.target_data.copy(), self.model_name, label="orig")
        
        for var_name in self.features:
            # Predict for perturbed feature data
            perturb_df = self.perturb_feature_by_feature_type(
                self.target_data.copy(), var_name, baseline, set_feature_values, self.feature_type)
            perturb_df = self.model_predict_probs_and_logits(perturb_df, self.model_name, label="perturb")
            
            # Odds ratios
            logit_orig = base_df['logit_orig']
            logit_perturb = perturb_df['logit_perturb']
            logit_diff = self.logit_score_comparison(logit_orig, logit_perturb)
            # odds_ratio = np.exp(logit_diff)
            # store
            prob_deltas_per_cell[var_name] = logit_diff
            prob_deltas_per_cell[f'perturb_{var_name}_prob'] = perturb_df['probs_perturb']
            prob_deltas_per_cell['orig_prob'] = base_df['probs_orig']

        return prob_deltas_per_cell.astype(float)
    
    def perturb_feature_by_feature_type(self, df, var_name, baseline, set_feature_values, feature_type):
        # perturb to baseline conditions
        if feature_type == "numerical": 
            return self.perturb_numerical(df, var_name, baseline, set_feature_values)
        elif feature_type == "binary": 
            return self.perturb_binary(df, var_name, baseline)
        elif feature_type == "categorical": 
            return self.perturb_categorical(df, var_name, baseline)
        raise
    
    def perturb_categorical(self, df, var_name, baseline, delimiter="__cat__"):
        assert baseline == 1.0, "Baseline always 1.0 for categorical features"
        perturbed = df.copy()
        perturbed[var_name] = baseline
        # Get other similar categories to reassign 0 value
        cat_name = var_name.split(delimiter)[0]
        similar_cats = [col for col in perturbed.columns if delimiter in col and col.split(delimiter)[0] == cat_name]
        for sim_cat in similar_cats:
            if sim_cat != var_name:
                perturbed[sim_cat] = 0.0
        return perturbed

    def perturb_binary(self, df, var_name, baseline):
        assert baseline in [0.0, 1.0], "Baseline either 1.0 or 0.0 for binary features"
        perturbed = df.copy()
        perturbed[var_name] = baseline
        return perturbed

    def perturb_numerical(self, df, var_name, baseline, set_feature_values=None):
        perturbed = df.copy()
        if set_feature_values and var_name in set_feature_values:
            set_val = set_feature_values[var_name]
        else:
            set_val = 0.0
        curr_val = perturbed.loc[:, var_name]
        pert_dist = curr_val - set_val
        new_val = self.perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline)
        perturbed[var_name] = new_val
        return perturbed
