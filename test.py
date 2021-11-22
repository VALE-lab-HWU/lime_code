import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from itertools import islice

import data_helper as dh
import model_helper as mh
import ml_helper as mlh

import pickle

RANDOM_SEED = 42


def save_pkl(res, fname='./res.pkl'):
    with open(fname, 'wb') as f:
        pickle.dump(res, f)


def load_pkl(fname='./res.pkl'):
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    return res


def cv_model_on_set(X, y, pipeline, model, args):
    return mlh.cross_validate(mh.run_model, X, y, k=3,
                              save_model=True,
                              model_fn=pipeline,
                              model=GridSearchCV,
                              model_kwargs={'estimator': model,
                                            'param_grid': args})


def cv_all_model_on_set(X, y, pipelines, models, args, name):
    res = {}
    for i in range(len(pipelines)):
        predict = cv_model_on_set(X, y, pipelines[i], models[i], args[i])
        res[name[i]] = predict
    return res


# return all lifetime images
def get_set_lf(it, lf, lb, patient, band):
    X = lf
    y = lb
    p = patient
    return X, y, p


# return all intensity images
def get_set_it(it, lf, lb, patient, band):
    X = it
    y = lb
    p = patient
    return X, y, p


# return all lifetime images of band 1
def get_set_lf_b1(it, lf, lb, patient, band):
    X = lf[band == 1]
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


# return all lifetime images of band 2
def get_set_lf_b2(it, lf, lb, patient, band):
    X = lf[band == 2]
    y = lb[band == 2]
    p = patient[band == 2]
    return X, y, p


# return all intensity images of band 1
def get_set_it_b1(it, lf, lb, patient, band):
    X = it[band == 1]
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


# return all intensity images of band 2
def get_set_it_b2(it, lf, lb, patient, band):
    X = it[band == 2]
    y = lb[band == 2]
    p = patient[band == 2]
    return X, y, p


# return lifetime of band 1 and 2 stacked
def get_set_lf_b1_b2(it, lf, lb, patient, band):
    X = np.concatenate((lf[band == 1], lf[band == 2]), axis=1)
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_b1_b2(it, lf, lb, patient, band):
    X = np.concatenate((it[band == 1], it[band == 2]), axis=1)
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b1(it, lf, lb, patient, band):
    X = np.concatenate((it[band == 1], lf[band == 1]), axis=1)
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b2(it, lf, lb, patient, band):
    X = np.concatenate((it[band == 2], lf[band == 2]), axis=1)
    y = lb[band == 2]
    p = patient[band == 2]
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b1_b2(it, lf, lb, patient, band):
    X = np.concatenate(
        (it[band == 1], it[band == 2], lf[band == 1], lf[band == 2]), axis=1)
    y = lb[band == 1]
    p = patient[band == 1]
    return X, y, p


def cv_one_set(fn, it, lf, lb, patient, band, pipelines, models, args, names):
    X, y, p = fn(it, lf, lb, patient, band)
    return {'X': X, 'y': y, 'p': p,
            **cv_all_model_on_set(X, y, pipelines, models, args, names)}


def cv_all_set(it, lf, lb, patient, band, pipelines, models, args, names):
    res = {}
    fn_dict = {'lf': get_set_lf, 'it': get_set_it,
               'lf_b1': get_set_lf_b1, 'lf_b2': get_set_lf_b2,
               'it_b1': get_set_it_b1, 'it_b2': get_set_it_b2,
               'lf_b1_b2': get_set_lf_b1_b2,
               'it_b1_b2': get_set_it_b1_b2,
               'it_lf_b1': get_set_it_lf_b1,
               'it_lf_b2': get_set_it_lf_b2,
               'it_lf_b1_b2': get_set_it_lf_b1_b2}
    for k in fn_dict:
        print(k)
        res[k] = cv_one_set(fn_dict[k], it, lf, lb, patient, band,
                            pipelines, models, args, names)
        save_pkl(res)
    return res


def get_test(it, lf, lb, p, b):
    idx = p == p[-1]
    itt, lft, lbt, pt, bt = it[idx], lf[idx], lb[idx], p[idx], b[idx]
    it2, lf2, lb2, p2, b2 = it[~idx], lf[~idx], lb[~idx], p[~idx], b[~idx]
    stk = StratifiedKFold(10)
    split = stk.split(it2, p2)
    idx2 = list(islice(split, 1))[0]
    itt = np.concatenate((itt, it2[idx2[1]]))
    lft = np.concatenate((lft, lf2[idx2[1]]))
    lbt = np.concatenate((lbt, lb2[idx2[1]]))
    pt = np.concatenate((pt, p2[idx2[1]]))
    bt = np.concatenate((bt, b2[idx2[1]]))
    return (it2[idx2[0]], lf2[idx2[0]], lb2[idx2[0]],
            p2[idx2[0]], b2[idx2[0]]), \
        (itt, lft, lbt, pt, bt)


def main(path=dh.PATH_CLEANED, filename=dh.FILENAME):
    (it, lf), label, patient, band = dh.get_data_complete(
        path, filename, all_feature=True)
    train, test = get_test(it, lf, label, patient, band)
    it, lf, lb, patient, band = train
    save_pkl([train, test], 'data.pkl')
    names = ['mlp', 'rf', 'svc', 'gpc', 'knn']
    # it's the same, but it's in case we don't want to do pca
    # or want to run specific process for some models
    pipelines = [mlh.build_pipeline_pca_model,
                 mlh.build_pipeline_pca_model,
                 mlh.build_pipeline_pca_model,
                 mlh.build_pipeline_pca_model,
                 mlh.build_pipeline_pca_model]
    models = [mh.build_mlp_model(max_iter=1000),
              mh.build_random_forest_model(
                  n_jobs=-1, max_depth=None, min_samples_split=2),
              mh.build_svc_model(probability=True),
              mh.build_gaussian_cla_model(
                  n_jobs=-1, max_iter_predict=500),
              mh.build_knn_model(n_jobs=-1)]
    args = [{'alpha': [1, 0.1, 0.001, 0.001, 0.0001],
             'hidden_layer_sizes': [(64, 32), (128, 64),
                                    (256, 64), (32, 16, 8)]},
            {'n_estimators': [100, 500, 1000, 2500, 5000],
             'max_features': [0.1, 0.25, 0.5, 0.75, 0.9]},
            {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'kernel': [[RBF(i) for i in np.logspace(-1, 1, 2)]]},
            {'n_neighbors': np.arange(1, 7)}]
    res = cv_all_set(it, lf, lb, patient, band,
                     pipelines, models, args, names)
    save_pkl([train, test], 'data.pkl')
    save_pkl(res)


if __name__ == '__main__':
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    main()
