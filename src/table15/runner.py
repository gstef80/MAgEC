from . import magec_utils as mg
from . import pima_utils as pm
from . import pipeline_utils as pu


def run(configs_path='../configs/pima_diabetes.yaml'):
    configs = pu.yaml_parser(configs_path)
    diabs_path = pu.get_from_configs(configs, 'DIABS_PATH')

    pima, x_train, x_validation, stsc, x_train_p, x_validation_p, y_train_p, y_validation_p = pm.pima_data(configs)
    print(x_train_p.shape)
    print(y_train_p.shape)

    models = pm.pima_models(x_train_p, y_train_p)

    mlp = models['mlp']
    rf = models['rf']
    lr = models['lr']
    ensemble = models['ensemble']

    # MLP
    case_mlp = mg.case_magecs(mlp, x_validation_p, model_name='mlp')
    magecs_mlp = mg.normalize_magecs(case_mlp, features=None, model_name='mlp')
    magecs_mlp = magecs_mlp.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    # RF
    case_rf = mg.case_magecs(rf, x_validation_p, model_name='rf')
    magecs_rf = mg.normalize_magecs(case_rf, features=None, model_name='rf')
    magecs_rf = magecs_rf.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    # LR
    case_lr = mg.case_magecs(lr, x_validation_p, model_name='lr')
    magecs_lr = mg.normalize_magecs(case_lr, features=None, model_name='lr')
    magecs_lr = magecs_lr.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)
    # ENSEMBLE
    case_en = mg.case_magecs(ensemble, x_validation_p, model_name='ensemble')
    magecs_en = mg.normalize_magecs(case_en, features=None, model_name='ensemble')
    magecs_en = magecs_en.merge(y_validation_p, left_on=['case', 'timepoint'], right_index=True)

    features =  pu.get_from_configs(configs, 'FEATURES')

    joined = mg.magec_models(magecs_mlp, 
                         magecs_rf, 
                         magecs_lr, 
                         magecs_en,
                         Xdata=x_validation_p, 
                         Ydata=y_validation_p, 
                         features=features)

    models = ('mlp', 'rf', 'lr')

    magec_totals = mg.avg_magecs(joined)

    print(magec_totals)

    # ranks = mg.magec_rank(joined, rank=len(features), features=features)
    # consensus = mg.magec_consensus(ranks, use_weights=True, models=models)

    # print(consensus.head())

    return magec_totals
