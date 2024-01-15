import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from itertools import islice
from arg import parse_args

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


def write_log(flog, msg):
    with open('./'+flog, 'a') as f:
        f.write(msg+'\n')


def reset_files(args):
    with open('./'+args.name, 'w+') as f:
        f.write('')
    with open('./'+args.log, 'w+') as f:
        f.write('')


# not using sklearn leave one group out
# cause idk how to give the argument of the splitter, p, to it
def run_model_on_set(X, y, p, pipeline, model, args):
    idx = np.unique(p, return_index=True)[1]
    # ugly one liner. Create a cv, each folds is (train, test), with one
    # patient as test
    cv = [(np.concatenate([range(0, idx[i]),
                           range(idx[i+1], idx[-1])]).astype(int),
           np.array(range(idx[i], idx[i+1]))) for i in range(len(idx)-1)]
    return mh.get_model(X, y, model_fn=pipeline, model=GridSearchCV,
                        model_kwargs={'estimator': model, 'param_grid': args,
                                      cv: cv})


def cv_all_model_on_set(X, y, p, pipelines, models, args, name, log):
    res = {}
    write_log(log, 'start!')
    for i in range(len(pipelines)):
        write_log(log, name[i])
        model = run_model_on_set(X, y, p, pipelines[i], models[i], args[i])
        res[name[i]] = model
    return res


def get_set_it_lf(train_b1, train_b2):
    Xb1 = np.concatenate((train_b1['it'], train_b1['lf']), axis=1)
    Xb2 = np.concatenate((train_b2['it'], train_b2['lf']), axis=1)
    X = np.concatenate((Xb1, Xb2))
    y = np.concatenate((train_b1['lb'], train_b2['lb']))
    p = np.concatenate((train_b1['p'], train_b2['p']))
    return X, y, p


# return all lifetime images
def get_set_lf(train_b1, train_b2):
    X = np.concatenate((train_b1['lf'], train_b2['lf']))
    y = np.concatenate((train_b1['lb'], train_b2['lb']))
    p = np.concatenate((train_b1['p'], train_b2['p']))
    return X, y, p


# return all intensity images
def get_set_it(train_b1, train_b2):
    X = np.concatenate((train_b1['it'], train_b2['it']))
    y = np.concatenate((train_b1['lb'], train_b2['lb']))
    p = np.concatenate((train_b1['p'], train_b2['p']))
    return X, y, p


# return all lifetime images of band 1
def get_set_lf_b1(train_b1, train_b2):
    X = train_b1['lf']
    y = train_b1['lb']
    p = train_b1['p']
    return X, y, p


# return all lifetime images of band 2
def get_set_lf_b2(train_b1, train_b2):
    X = train_b2['lf']
    y = train_b2['lb']
    p = train_b2['p']
    return X, y, p


# return all intensity images of band 1
def get_set_it_b1(train_b1, train_b2):
    X = train_b1['it']
    y = train_b1['lb']
    p = train_b1['p']
    return X, y, p


# return all intensity images of band 2
def get_set_it_b2(train_b1, train_b2):
    X = train_b2['it']
    y = train_b2['lb']
    p = train_b2['p']
    return X, y, p


# return lifetime of band 1 and 2 stacked
def get_set_lf_b1_b2(train_b1, train_b2):
    X = np.concatenate((train_b1['lf'], train_b2['lf']), axis=1)
    y = train_b1['lb']
    p = train_b1['p']
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_b1_b2(train_b1, train_b2):
    X = np.concatenate((train_b1['it'], train_b2['it']), axis=1)
    y = train_b1['lb']
    p = train_b1['p']
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b1(train_b1, train_b2):
    X = np.concatenate((train_b1['it'], train_b1['lf']), axis=1)
    y = train_b1['lb']
    p = train_b1['p']
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b2(train_b1, train_b2):
    X = np.concatenate((train_b2['it'], train_b2['lf']), axis=1)
    y = train_b2['lb']
    p = train_b2['p']
    return X, y, p


# return intensity of band 1 and 2 stacked
def get_set_it_lf_b1_b2(train_b1, train_b2):
    X = np.concatenate((train_b1['it'], train_b1['lf'],
                        train_b2['it'], train_b2['lf']),
                       axis=1)
    y = train_b2['lb']
    p = train_b2['p']
    return X, y, p


def cv_one_set(
        fn, train_b1, train_b2, pipelines, models, args, names, log):
    X, y, p = fn(train_b1, train_b2)
    # idx = np.random.permutation(len(X))
    # X, y, p = X[idx], y[idx], p[idx]
    return {'X': X, 'y': y, 'p': p,
            **cv_all_model_on_set(X, y, p, pipelines, models,
                                  args, names, log)}


def cv_all_set(
        train_b1, train_b2, pipelines, models, args, names, g_args):
    res = {}
    fn_dict = {'lf': get_set_lf, 'it': get_set_it,
               'lf_b1': get_set_lf_b1, 'lf_b2': get_set_lf_b2,
               'it_b1': get_set_it_b1, 'it_b2': get_set_it_b2,
               'lf_b1_b2': get_set_lf_b1_b2,
               'it_b1_b2': get_set_it_b1_b2,
               'it_lf_b1': get_set_it_lf_b1,
               'it_lf_b2': get_set_it_lf_b2,
               'it_lf_b1_b2': get_set_it_lf_b1_b2}
    write_log(g_args.log, g_args.set)
    res = cv_one_set(fn_dict[g_args.set], train_b1, train_b2,
                     pipelines, models, args, names, log=g_args.log)
    save_pkl(res, fname='./'+g_args.name)
    # for k in fn_dict:
    #     print(k)
    #     res[k] = cv_one_set(fn_dict[k], it, lf, lb, patient, band,
    #                         pipelines, models, args, names)
    #     save_pkl(res)
    return res


def get_idx_train_test(p):
    idx_test = p == p[-1]
    idx_train = ~idx_test
    return np.nonzero(idx_train)[0], np.nonzero(idx_test)[0]


def get_idx_b1(b):
    return np.nonzero(b == 1)[0]


def get_test(it, lf, lb, p, b):
    args = locals()
    idxb1 = get_idx_b1(b)
    # train_p, test_p = get_idx_train_test(p[idxb1])
    # stk = StratifiedKFold(10, shuffle=True)
    # train, test_b = list(islice(stk.split(train_p, p[train_p]), 1))[0]
    # test = np.concatenate((test_b, test_p))
    idxs = [idxb1, idxb1+1]
    return idxs


def main(global_args, path=dh.PATH_CLEANED, filename=dh.FILENAME):
    write_log(global_args.log, 'read data')
    (it, lf), label, patient, band = dh.get_data_complete(
        path, filename, all_feature=True)
    write_log(global_args.log, 'split test')
    train_b1, train_b2 = get_test(
        it, lf, label, patient, band)
    write_log(global_args.log, 'save data')
    save_pkl([train_b1, train_b2],
             global_args.set+'_data.pkl')
    # names = ['mlp', 'rf', 'svc', 'knn']
    #
    # it's the same, but it's in case we don't want to do pca
    # or want to run specific process for some models
    pipelines = {'mlp': mlh.build_pipeline_pca_model,
                 'rf': mlh.build_pipeline_pca_model,
                 'svc': mlh.build_pipeline_pca_model,
                 'knn': mlh.build_pipeline_pca_model}
    # pipelines = [mlh.build_pipeline_pca_model]
    # models = [mh.build_svc_model(probability=False)]
    models = {'mlp': mh.build_mlp_model(max_iter=1000),
              'rf': mh.build_random_forest_model(
                  n_jobs=-1, max_depth=None, min_samples_split=2),
              'svc': mh.build_svc_model(probability=False),
              'knn': mh.build_knn_model(n_jobs=-1)}
    args = {'mlp': {'alpha': [1, 0.1, 0.001, 0.001, 0.0001],
                    'hidden_layer_sizes': [(64, 32), (128, 64),
                                           (256, 64), (32, 16, 8)]},
            'rf': {'n_estimators': [100, 500, 1000, 2500, 5000],
                   'max_features': [0.1, 0.25, 0.5, 0.75, 0.9]},
            'svc': {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
            'knn': {'n_neighbors': [1, 5, 10, 15, 20, 25]}}
    # args = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #          'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}]
    name = [global_args.model]
    pipeline = [pipelines[global_args.model]]
    model = [models[global_args.model]]
    arg = [args[global_args.model]]
    res = cv_all_set(train_b1, train_b2,
                     pipeline, model, arg, name, global_args)
    save_pkl(res)


if __name__ == '__main__':
    args = parse_args()
    reset_files(args)
    write_log(args.log, 'Set seed')
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    main(args)
