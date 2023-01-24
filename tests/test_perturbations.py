from src.table15.utils.magec_utils import *
from src.table15.utils.models_container import ModelsContainer
from src.table15.utils import pipeline_utils as plutils
from src.table15.utils.data_tables import DataTables
from src.table15.models.test_linear_model import TestBasicModel

import pytest

@ pytest.fixture
def z_pert():
    return Z_Perturbation()

@pytest.fixture
def data_and_model():
    yaml_path = "/Users/ag46548/dev/github/KaleRP/table15/tests/configs/t_configs.yaml"
    configs = plutils.yaml_parser(yaml_path)
    
    dutils = DataTables().generate_data(configs)
    mutils = ModelsContainer(dutils.x_train_p, dutils.y_train_p, dutils.x_validation_p)
    return dutils, mutils


@pytest.fixture
def probs_data():
    probs = [0.25, 0.5, 0.75]
    return probs


@pytest.fixture
def numerical_perturbation_df():
    d1 = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    d2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    df = pd.DataFrame(zip(d1, d2), columns=["a", "b"])
    return df


@pytest.fixture
def simple_data():
    data = np.ones(10)
    df = pd.DataFrame(data, columns=["a"])
    return df
    

def test_static_prediction(simple_data, z_pert):
    model = TestBasicModel()
    test_data = simple_data
    test_df = z_pert.model_predict_probs_and_logits(test_data, model, label="orig")
    test_df = test_df.head(1)
    expected_data = [[0.5, 0.0]]
    expected_df = pd.DataFrame(expected_data, columns=["probs_orig", "logit_orig"])
    pd.testing.assert_frame_equal(test_df, expected_df)
    


def test_perturb_categorical(z_pert):
    data = [[1, 0, 0 ,0], [1, 0, 0 ,0], [0, 0, 1 ,0], [0, 0, 0 ,1]]
    columns = ["A__cat__red", "A__cat__yellow", "A__cat__green", "A__cat__blue"]
    var_name = "A__cat__yellow"
    baseline = 1.0
    
    df = pd.DataFrame(data, columns=columns, dtype=np.float64).reset_index()
    df = z_pert.perturb_categorical(df, var_name, baseline)
    expected_data = [[0, 1, 0 ,0], [0, 1, 0 ,0], [0, 1, 0 ,0], [0, 1, 0 ,0]]
    expected_df = pd.DataFrame(expected_data, columns=columns, dtype=np.float64).reset_index()
    pd.testing.assert_frame_equal(df, expected_df)


def test_perturb_binary(z_pert):
    data = [[1, 1], [0, 0], [0, 1], [1, 0]]
    columns = ["is_foo", "is_bar"]
    var_name = ["is_foo"]
    baseline = 1.0
    df = pd.DataFrame(data, columns=columns, dtype=np.float64).reset_index()
    test_df = z_pert.perturb_binary(df, var_name, baseline)
    expected_data = [[1, 1], [1, 0], [1, 1], [1, 0]]
    expected_df = pd.DataFrame(expected_data, columns=columns, dtype=np.float64).reset_index()
    pd.testing.assert_frame_equal(test_df, expected_df)

    var_name = "is_bar"
    df = pd.DataFrame(data, columns=columns, dtype=np.float64).reset_index()
    test_df = z_pert.perturb_binary(df, var_name, baseline)
    expected_data = [[1, 1], [0, 1], [0, 1], [1, 1]]
    expected_df = pd.DataFrame(expected_data, columns=columns, dtype=np.float64).reset_index()
    pd.testing.assert_frame_equal(test_df, expected_df)
    

def test_perturb_numerical(numerical_perturbation_df):
    var_name = "a"
    baseline = 0.3
    set_feature_values = None
    test_df = perturb_numerical(numerical_perturbation_df, var_name, baseline, set_feature_values=set_feature_values)
    expected_data = zip([-1.4, -0.7, -0.35, 0.0, 0.35, 0.7, 1.05], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_df = pd.DataFrame(expected_data, columns=["a", "b"], dtype=np.float64)
    pd.testing.assert_frame_equal(test_df, expected_df)
    
    set_feature_values = {"a": 1.0, "b": -2.0}
    var_name = "a"
    test_df = perturb_numerical(numerical_perturbation_df, var_name, baseline, set_feature_values=set_feature_values)
    expected_data = zip([-1.1, -0.4, -0.05, 0.3, 0.65, 1.0, 1.35], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_df = pd.DataFrame(expected_data, columns=["a", "b"], dtype=np.float64)
    pd.testing.assert_frame_equal(test_df, expected_df)
    
    var_name = "b"
    test_df = perturb_numerical(numerical_perturbation_df, var_name, baseline, set_feature_values=set_feature_values)
    expected_data = zip([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
    expected_df = pd.DataFrame(expected_data, columns=["a", "b"], dtype=np.float64)
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_perturb_num_series_with_baseline_scaling(numerical_perturbation_df):
    curr_val = numerical_perturbation_df["a"]
    
    set_val = 0.0
    baseline_scaling = 1.0
    pert_dist = curr_val - set_val
    test_series = perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline_scaling)
    expected_series = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name="a")
    pd.testing.assert_series_equal(test_series, expected_series)
    
    baseline_scaling = 0.3
    test_series = perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline_scaling)
    expected_series = pd.Series([-1.4, -0.7, -0.35, 0.0, 0.35, 0.7, 1.05], name="a")
    pd.testing.assert_series_equal(test_series, expected_series)
    
    set_val = -0.5
    baseline_scaling = 1.0
    pert_dist = curr_val - set_val
    test_series = perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline_scaling)
    expected_series = pd.Series([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5], name="a")
    pd.testing.assert_series_equal(test_series, expected_series)
    
    set_val = -0.5
    baseline_scaling = 0.3
    pert_dist = curr_val - set_val
    test_series = perturb_num_series_with_baseline_scaling(curr_val, pert_dist, baseline_scaling)
    expected_series = pd.Series([-1.55, -0.85, -0.5, -0.15, 0.2, 0.55, 0.9], name="a")
    pd.testing.assert_series_equal(test_series, expected_series)
    

def test_get_logit_ln(probs_data):
    expected_result = -1.099
    result = get_logit_ln(probs_data[0])
    result = round(result, 3)
    
    assert result == expected_result
    expected_result = 0
    result = get_logit_ln(probs_data[1])
    assert result == expected_result
    
    expected_result = 1.099
    result = get_logit_ln(probs_data[2])
    result = round(result, 3)
    assert result == expected_result
    

def test_get_logit_base_2(probs_data):
    expected_result = -1.585
    result = get_logit_base2(probs_data[0])
    result = round(result, 3)
    assert result == expected_result
    
    expected_result = 0
    result = get_logit_base2(probs_data[1])
    assert result == expected_result
    
    expected_result = 1.585
    result = get_logit_base2(probs_data[2])
    result = round(result, 3)
    assert result == expected_result
    