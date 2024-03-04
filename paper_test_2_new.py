import sys

import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from functools import partial
from arg import parse_args

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


def run_on_one_fold(ds, fold, model_fn):
    data = load_pkl(f'pca/pca_{ds}_{fold}.pkl')
    model = model_fn['model'](**model_fn['kwargs'])
    model.fit(data['train']['X'], data['train']['y'])
    pred = model.predict(data['test']['X'])
    return pred, data['test']['y']


def run_all_fold(ds, model_fn, fold_nb=11):
    y_pred = []
    y_true = []
    for i in range(fold_nb):
        pr, tr = run_on_one_fold(ds, i, model_fn)
        y_pred.append(pr)
        y_true.append(tr)
    return y_pred, y_true


def build_model(model):
    models = {
        'mlp': {
            'model': mh.build_mlp_model,
            'kwargs': {
                'alpha': 0.01,
                'hidden_layer_sizes':
                (128, 64),
                'max_iter': 1500
            }
        },
        'rf': {
            'model': mh.build_random_forest_model,
            'kwargs': {
                'max_features': 0.1,
                'n_estimators': 500,
            }
        },
        'knn': {
            'model': mh.build_knn_model,
            'kwargs': {
                'n_neighbors': 1
            }
        },
        'svm': {
            'model': mh.build_svc_model,
            'kwargs': {
                'C': 100,
                'gamma': 0.001,
                'probability': False
            }
        }
    }
    return models[model]


def main(global_args):
    dset = global_args.set
    model_fn = build_model(args.model)
    y_pred, y_true = run_all_fold(dset, model_fn)
    for i in range(len(y_pred)):
        with open(f'{global_args.model}_{dset}_{i}.txt', 'w') as f:
            mlh.compare_class(y_pred[i], y_true[i], verbose=2,
                              f=f, unique_l=[1, 0])
    y_true = [j for i in y_true for j in i]
    y_pred = [j for i in y_pred for j in i]
    with open(f'{global_args.model}_{dset}_all.txt', 'w') as f:
        mlh.compare_class(y_pred, y_true, verbose=3, f=f, unique_l=[1, 0])


if __name__ == '__main__':
    args = parse_args()
    reset_files(args)
    write_log(args.log, 'Set seed')
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    main(args)
