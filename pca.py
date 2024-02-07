import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from itertools import islice
from arg import parse_args

import paper_test as pt
import data_helper as dh
import model_helper as mh
import ml_helper as mlh

import pickle

RANDOM_SEED = 42

def get_20_first_patient(it, lf, label, patient, band):
    res = []
    for i in np.unique(patient):
        res.extend(list(np.where(patient==i)[0][:20]))
    return it[res], lf[res], label[res], patient[res], band[res]


def main(path=dh.PATH_CLEANED, filename=dh.FILENAME):
    (it, lf), label, patient, band = dh.get_data_complete(
        path, filename, all_feature=True)
    # it, lf, label, patient, band = get_20_first_patient(it, lf, label, patient, band)
    train_b1, train_b2 = pt.get_test(
        it, lf, label, patient, band)
    fn_dict = {'lf': pt.get_set_lf, 'it': pt.get_set_it,
               'lf_b1': pt.get_set_lf_b1, 'lf_b2': pt.get_set_lf_b2,
               'it_b1': pt.get_set_it_b1, 'it_b2': pt.get_set_it_b2,
               'lf_b1_b2': pt.get_set_lf_b1_b2,
               'it_b1_b2': pt.get_set_it_b1_b2,
               'it_lf_b1': pt.get_set_it_lf_b1,
               'it_lf_b2': pt.get_set_it_lf_b2,
               'it_lf_b1_b2': pt.get_set_it_lf_b1_b2}
    for k, v in fn_dict.items():
        X, y, p = v(train_b1, train_b2)
        idx = np.unique(p)
        cv = [(np.where(p != i)[0], np.where(p == i)[0]) for i in idx]
        pca = mlh.build_pipeline_pca_scale()
        for patient, (train, test) in enumerate(cv):
            n_train = pca.fit_transform(X[train])
            n_test = pca.transform(X[test])
            res_train = {'X': n_train, 'y': y, 'p': p}
            res_test = {'X': n_test, 'y': y, 'p': p}
            pt.save_pkl({'train': res_train, 'test': res_test}, f'pca_{k}_{patient}.pkl')
    

if __name__ == '__main__':
    main()
