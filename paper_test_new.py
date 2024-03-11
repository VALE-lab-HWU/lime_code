import numpy as np
from sklearn.model_selection import ParameterGrid
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


def run_on_one_fold(ds, fold, model_fn, args_model):
    data = load_pkl(f'pca/pca_{ds}_{fold}.pkl')
    res = []
    for i, arg in enumerate(args_model):
        model = model_fn(**arg)
        model.fit(data['train']['X'], data['train']['y'])
        pred = model.predict(data['test']['X'])
        res.append((pred, data['test']['y']))
    return res


def run_all_fold(ds, model_fn, args_model, fold_nb=11):
    y_pred = [[] for i in range(len(args_model))]
    y_true = [[] for i in range(len(args_model))]
    for i in range(fold_nb):
        res_pr = run_on_one_fold(ds, i, model_fn, args_model)
        for j, (pr, tr) in enumerate(res_pr):
            y_pred[j].append(pr)
            y_true[j].append(tr)
    return y_pred, y_true


def build_model(model):
    models = {'mlp': partial(mh.build_mlp_model, max_iter=1000),
              'rf': partial(mh.build_random_forest_model,
                            n_jobs=-1, max_depth=None, min_samples_split=2),
              'svc': partial(mh.build_svc_model, gamma='scale',
                             probability=False),
              'knn': partial(mh.build_knn_model, n_jobs=-1)}
    return models[model]


def get_arg_model(model):
    args = {'mlp': {'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
                    'hidden_layer_sizes': [(64, 32), (128, 64),
                                           (256, 64), (32, 16, 8)]},
            'rf': {'n_estimators': [100, 500, 1000, 2500],
                   'max_features': [0.1, 0.25, 0.5, 0.75, 0.9]},
            'svc': {'C': [0.01, 0.1, 1, 10, 100]},
            'knn': {'n_neighbors': [1, 5, 10, 15, 20, 25]}}
    return ParameterGrid(args[model])


def main(global_args):
    seed = global_args.seed
    dset = global_args.set
    model_fn = build_model(args.model)
    args_model = get_arg_model(args.model)
    y_preds, y_trues = run_all_fold(dset, model_fn, args_model)
    with open(f'{global_args.model}_{dset}_seed_{seed}.pkl', 'w') as f:
        print(list(enumerate(args_model)), file=f)
    for i in range(len(y_preds)):
        for j in range(len(y_preds[i])):
            with open(f'{global_args.model}_{i}_{dset}_{j}_seed_{seed}.txt', 'w') as f:
                mlh.compare_class(y_preds[i][j], y_trues[i][j], verbose=2,
                                  f=f, unique_l=[1, 0])
    y_trues = [[j for i in y_true for j in i] for y_true in y_trues]
    y_preds = [[j for i in y_pred for j in i] for y_pred in y_preds]
    for i in range(len(y_preds)):
        with open(f'{global_args.model}_{i}_{dset}_all_seed_{seed}.txt', 'w') as f:
            mlh.compare_class(y_preds[i], y_trues[i], verbose=3,
                              f=f, unique_l=[1, 0])


if __name__ == '__main__':
    args = parse_args()
    reset_files(args)
    write_log(args.log, 'Set seed')
    print('Set seed')
    np.random.seed(args.seed)
    main(args)
