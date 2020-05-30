import numpy as np
import magec_utils as mg
import mimic_utils as mimic
from sklearn.utils import class_weight
import multiprocessing
from multiprocessing import set_start_method
import warnings


def get_magecs(return_dict, model, x_magec, model_name, baseline=None):
    p_name = multiprocessing.current_process().name
    print('Starting:', p_name)
    if model_name == 'lstm':
        magecs = mg.case_magecs(model, x_magec, model_name=model_name, baseline=baseline, timeseries=True)
    else:
        magecs = mg.case_magecs(model, x_magec, model_name=model_name, baseline=baseline)
    print('Magecs for {} computed...'.format(p_name))
    magecs = mg.normalize_magecs(magecs, features=None, model_name=model_name)
    print('Magecs for {} normalized...'.format(p_name))
    print('Exiting :', p_name)
    return_dict[p_name] = magecs


def main():
    # MIMIC-III
    print('getting mimic data...')
    df = mimic.get_mimic_data()
    # Data featurized using 'last' measurements
    print('featurizing mimic data...')
    df_ml = mimic.get_ml_data(df)
    # Data featurized as time-series
    df_time = mimic.get_ml_series_data(df)
    x_train, x_validation, stsc, xst_train, xst_validation, \
    Y_train, Y_validation = mimic.train_valid_ml(df_ml)
    # train/valid split
    stsc2, series_means, df_series_train, df_series_valid, xt_train, Yt_train, \
    xt_valid, Yt_valid = mimic.train_valid_series(df_time, Y_validation)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train['label']),
                                                      Y_train['label'])
    # train models
    print('training models...')
    models = mimic.mimic_models(xst_train, Y_train, xt_train, Yt_train, class_weights)

    xy_magec = df_series_valid.copy()
    x_magec_cols = list(set(xy_magec.columns) - {'label'})
    x_magec = xy_magec[x_magec_cols]

    print('getting magecs...')
    with multiprocessing.Manager() as manager:
        out = manager.dict()
        baselines = [None, 0.5, 0.7, 0.9]
        processes = []
        keys = []
        for model in models.keys():
            for baseline in baselines:
                key = model + '_p{}'.format(int(baseline * 10)) if baseline is not None else model + '_0'
                keys.append(key)
                clf = models[model]
                if model in ['mlp', 'lstm']:
                    clf = clf.model
                p = multiprocessing.Process(name=key,
                                            target=get_magecs,
                                            args=(out, clf, x_magec, model, baseline))
                processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        for key in keys:
            df = out[key]
            df.to_csv('magec_{}.csv'.format(key))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    set_start_method("spawn")
    main()
