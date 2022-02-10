import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from itertools import islice
from arg import parse_args

import paper_test

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


def take(a, idx):
    return [a[i] for i in idx]


def get_data_per_patient(patient, *arg):
    idx = [np.where(patient == i) for i in np.unique(patient)]
    return take(patient, idx), *[take(i, idx) for i in arg]


def run_cross_validation_custom(fns, datas, labels, shuffle=False):
    res = []
    for i in range(len(datas)):
        print('fold %d' % i)
        x_train = np.concatenate([*datas[:i], *datas[i+1:]],
                                 dtype=float)
        y_train = np.concatenate([*labels[:i], *labels[i+1:]],
                                 dtype=int)
        if shuffle:
            x_train, y_train = mlh.shuffle_arrays_of_array(x_train, y_train)
        x_test = datas[i]
        y_test = np.array(labels[i], dtype=int)
        # not sure random state works for this PCA
        pca = mlh.PCA(n_components=0.95, svd_solver='full', random_state=42)
        pca_x_train = pca.fit_transform(x_train)
        pca_x_test = pca.transform(x_test)
        predicted = {}
        for name in fns:
            predicted[name] = mh.run_model(
                pca_x_train, y_train, pca_x_test,
                model_fn=fns[name]['model'], **fns[name]['kwargs'])
        res.append((predicted, y_test))
    return res


def get_band_set(it, lf, lb, p, b):
    args = locals()
    return [{i: args[i][b == j] for i in args} for j in [1, 2]]


def main(global_args, path=dh.PATH_CLEANED, filename=dh.FILENAME):
    write_log(global_args.log, 'read data')
    (it, lf), label, patient, band = dh.get_data_complete(
        path, filename, all_feature=True)
    fn_dict = {'lf': paper_test.get_set_lf, 'it': paper_test.get_set_it,
               'lf_b1': paper_test.get_set_lf_b1,
               'lf_b2': paper_test.get_set_lf_b2,
               'it_b1': paper_test.get_set_it_b1,
               'it_b2': paper_test.get_set_it_b2,
               'lf_b1_b2': paper_test.get_set_lf_b1_b2,
               'it_b1_b2': paper_test.get_set_it_b1_b2,
               'it_lf_b1': paper_test.get_set_it_lf_b1,
               'it_lf_b2': paper_test.get_set_it_lf_b2,
               'it_lf_b1_b2': paper_test.get_set_it_lf_b1_b2}
    b1, b2 = get_band_set(it, lf, label, patient, band)
    X, y, p = fn_dict['it'](b1, b2)
    p, X, y = get_data_per_patient(p, X, y)
    # models pipeline pca with random state
    models = {
        'mlp': {
            'model': mh.build_mlp_model,
            'kwargs': {
                'alpha': 1,
                'hidden_layer_sizes':
                (256, 64),
                'max_iter': 1500
            }
        },
        'rf': {
            'model': mh.build_random_forest_model,
            'kwargs': {
                'max_features': 0.1
            }
        },
        'knn': {
            'model': mh.build_knn_model,
            'kwargs': {
                'n_neighbors': 20
            }
        },
        'svm': {
            'model': mh.build_svc_model,
            'kwargs': {
                'C': 10,
                'gamma': 0.001
            }
        }
    }
    res = run_cross_validation_custom(models, X, y, shuffle=False)
    save_pkl(res, global_args.name)


if __name__ == '__main__':
    args = parse_args()
    reset_files(args)
    write_log(args.log, 'Set seed')
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    main(args)
