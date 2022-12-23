from src.table15.utils.magec_utils import *

import pytest


@pytest.fixture
def probs_data():
    probs = [0.25, 0.5, 0.75]
    return probs




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
    
    
    
