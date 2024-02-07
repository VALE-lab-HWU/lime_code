import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
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


def run_on_one_fold(ds, fold, model_fn):
    data = load_pkl(f'pca_{ds}_{fold}.pkl')
    model = model_fn()
    model.fit(data['train']['X'])
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
    

def build_model():
    pass


def main(global_args, path=dh.PATH_CLEANED, filename=dh.FILENAME):
    dset = global_args.set
    model_fn = build_model()
    metric_fn = get_metric(global_args.metric)
    y_pred, y_true = run_on_one_fold(dset, model_fn)
    metric = {}
    for i in range(len(y_pred)):
        metric[i] = metric_fn(y_true, y_pred)
    y_true = [j for i in y_true for j in i]
    y_pred = [j for i in y_pred for j in i]
    metric['all'] = metric_fn(y_true, y_pred)
    save_pkl(metric, f'res_{global_args.ensemble}_{dset}.pkl')
    

if __name__ == '__main__':
    args = parse_args()
    reset_files(args)
    write_log(args.log, 'Set seed')
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    main(args)
