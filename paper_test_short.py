import numpy as np
import paper_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import data_helper as dh
import model_helper as mh
import ml_helper as mlh

import pickle


RANDOM_SEED = 42


def save_pkl(res, fname='./res.pkl'):
    with open(fname, 'wb') as f:
        pickle.dump(res, f)


def get_band_set(it, lf, lb, p, b):
    args = locals()
    return [{i: args[i][b == j] for i in args} for j in [1, 2]]


def main(path=dh.PATH_CLEANED, filename=dh.FILENAME):
    (it, lf), label, patient, band = dh.get_data_complete(
        path, filename, all_feature=True)
    b1, b2 = get_band_set(it, lf, label, patient, band)
    X, y, p = paper_test.get_set_it(b1, b2)
    x1, x2, y1, y2 = train_test_split(X, y, test_size=0.3, shuffle=True,
                                      random_state=42)
    pca = mlh.PCA(n_components=0.95, svd_solver='full', random_state=42)
    px1 = pca.fit_transform(x1)
    px2 = pca.transform(x2)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(px1, y1)
    out = knn.predict(px2)
    save_pkl({'x2': x2, 'px2': px2, 'y2': y2, 'out': out},
             fname='./out_knn.pkl')
    mlh.compare_class(out, y2, verbose=3, L=12, unique_l=[1, 0])


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    main()
