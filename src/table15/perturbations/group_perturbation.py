from typing import List
from src.table15.perturbations.perturbation import Perturbation

import pandas as pd


class GroupPerturbation(Perturbation):
    def __init__(self, target_data: pd.DataFrame, model_name: str, features: List[str], feature_type: str) -> None:
        super().__init__(target_data, model_name, features, feature_type)
    
    def run_perturbation(self, set_feature_values, baseline=1):
        prob_deltas_per_cell = pd.DataFrame(index=self.target_data.index, columns=self.target_data.columns)
        
        # Predict for original data
        base_df = self.model_predict_probs_and_logits(self.target_data.copy(), self.model_name, label="orig")
        
        for grouped_vars in self.features:
            perturb_df = self.perturb_feature_by_feature_type(
                self.target_data.copy(), grouped_vars, baseline, set_feature_values, self.feature_type)
            perturb_df = self.model_predict_probs_and_logits(perturb_df, self.model_name, label="perturb")

            # Odds ratios
            logit_orig = base_df['logit_orig']
            logit_perturb = perturb_df['logit_perturb']
            logit_diff = self.logit_score_comparison(logit_orig, logit_perturb)
            # odds_ratio = np.exp(logit_diff)
            # store
            grouped_vars_col = "::".join(grouped_vars)
            prob_deltas_per_cell[grouped_vars_col] = logit_diff
            prob_deltas_per_cell[f'perturb_{grouped_vars_col}_prob'] = perturb_df['probs_perturb']
            prob_deltas_per_cell['orig_prob'] = base_df['probs_orig']
    
        return prob_deltas_per_cell.astype(float)
    
    def perturb_group(self, df, grouped_vars, baseline):
        perturbed = df.copy()
        
        set_val = 0.0
        curr_vals = perturbed.loc[:, grouped_vars]
        new_vals = self.perturb_num_df_with_baseline_scaling(curr_vals, baseline)
        perturbed[grouped_vars] = new_vals
        return perturbed
        
    def perturb_num_df_with_baseline_scaling(self, curr_vals, baseline_scaling):
        return curr_vals - float(baseline_scaling)
    
    def perturb_feature_by_feature_type(self, df, var_name, baseline, set_feature_values, feature_type):
        if feature_type == "grouped": 
            return self.perturb_group(df, var_name, baseline)
        raise