from table15.magec_utils import *

def test_get_logit_base_2():
    args = {"prob" : 0.75}
    print(args["prob"])
    obs = get_logit(args["prob"])
    assert round(obs, 3) == 1.585
