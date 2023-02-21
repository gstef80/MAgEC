from typing import List
import pandas as pd
import numpy as np

from src.table15.models.model import Model

class Perturbation:
    def __init__(self, target_data: pd.DataFrame, model: Model, features: List[str], feature_type:str) -> None:
        self.target_data = target_data
        self.model = model
        self.features = features
        self.feature_type = feature_type
        self.perform_data_checks(self.target_data, self.features, self.feature_type)
    
    def perform_data_checks(self, target_data, features, feature_type):
        assert 'case' in target_data.index.names, "missing 'case' from index"
        
        features = np.asarray(features)
        assert len(features) > 0, f"No features here to perturb. Feature type: {feature_type}."
        
        if feature_type != "grouped":
            # assert features only have 2 values
            binary = target_data[features].apply(lambda x: len(np.unique(x)), ) <= 2
            binary = binary[binary].index.tolist()
            assert (feature_type in ["binary", "categorical"] and len(binary) > 0) or (feature_type not in ["binary", "categorical"] and len(binary) == 0), (
                f"Mismatch between binary feature_type = {feature_type} and len(binary) = {len(binary)}")

            epsilons = dict()
            for var_name in features:
                if var_name in binary:
                    epsilon = target_data[var_name].unique().tolist()
                    epsilons[var_name] = epsilon
                    if type(epsilons[var_name]) is list and len(epsilons[var_name]) <= 2:
                        assert feature_type in ["binary", "categorical"]
                    if "__cat__" in var_name:
                        assert feature_type == "categorical"
                else:    
                    assert var_name not in epsilons
                    assert feature_type == "numerical"


    def model_predict_probs_and_logits(self, df, model, label=None):
        probs = model.predict(df)
        logits = self.get_logits_ln(probs)
        df_cols = df.columns
        df['probs_{}'.format(label)] = probs
        df['logits_{}'.format(label)] = logits
        df = df.drop(df_cols, axis=1)
        return df
        
    def get_logits_base2(self, prob, eps=1e-16):
        return np.log2((prob+eps)/(1-prob+eps))

    def get_logits_ln(self, prob, eps=1e-16):
        return np.log((prob+eps)/(1-prob+eps))
    
    def logits_score_comparison(self, x_orig, x_perturb):
        return x_perturb - x_orig
    
    def relative_risk_comparison(self, probs_orig, probs_perturb):
        return ((probs_orig - probs_perturb) / probs_orig) * 100
    
    def logits_diff_to_odds_ratio(self, logits_diff):
        return np.exp(logits_diff)

    def perturb_num_series_with_baseline_scaling(self, curr_val, perturbation_distance, baseline_scaling):
        return curr_val - (perturbation_distance * float(baseline_scaling))
    
    def calculate_output(self, base_df, perturb_df, output_type):
        if output_type == "relative_risk":
            probs_orig = base_df["probs_orig"]
            probs_perturb = perturb_df[f"probs_perturb"]
            output = self.relative_risk_comparison(probs_orig, probs_perturb)
            
        elif output_type in ["logits", "odds_ratios"]:
            logits_orig = base_df["logits_orig"]
            logits_perturb = perturb_df["logits_perturb"]
            logits_diff = self.logits_score_comparison(logits_orig, logits_perturb)
            if output_type == "logits":
                output = logits_diff
            elif output_type == "odds_ratios":
                output = self.logits_diff_to_odds_ratio(logits_diff)
        
        return output
    
    def store_outputs(self, output_df, feature_col, output, base_df, perturb_df):
        output_df[feature_col] = output
        output_df[f'perturb_{feature_col}_prob'] = perturb_df['probs_perturb']
        output_df['orig_prob'] = base_df['probs_orig']
        return output_df
