from typing import List

import numpy as np
import pandas as pd

from src.table15.models.model import Model
from src.table15.perturbations.perturbation import Perturbation


class GroupPerturbation(Perturbation):
    def __init__(self, target_data: pd.DataFrame, model: Model, features: List[str], feature_type: str) -> None:
        super().__init__(target_data, model, features, feature_type)
    
    def run_perturbation(self, set_feature_values, output_type, baseline=1.0):
        output_df = pd.DataFrame(index=self.target_data.index, columns=self.target_data.columns)
        
        # Predict for original data
        base_df = self.model_predict_probs_and_logits(self.target_data.copy(), self.model, label="orig")
        
        for grouped_vars in self.features:
            # Predict for perturbed grouped features data
            perturb_df = self.perturb_feature_by_feature_type(
                self.target_data.copy(), grouped_vars, baseline, set_feature_values, self.feature_type)
            perturb_df = self.model_predict_probs_and_logits(perturb_df, self.model, label="perturb")
            output = self.calculate_output(base_df, perturb_df, output_type)
            grouped_vars_col = "::".join(grouped_vars)
            output_df = self.store_outputs(output_df, grouped_vars_col, output, base_df, perturb_df)
    
        return output_df.astype(float)
    
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
