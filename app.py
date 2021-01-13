import lime
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from helper import cross_validate

PATH = '../data'
FILENAME = 'all_patient.pickle'
RANDOM_SEED = 42


def read_data(filepath=PATH+'/'+FILENAME, dict=False):
    data = pd.read_pickle(filepath)
    if dict:
        data = [data[i] for i in data][0]
    return data


def concat_data(path=PATH):
    listdir = os.listdir(path)
    listdir.sort()
    res = pd.DataFrame()
    r = []
    for folder in listdir:
        if folder[0] == '2':
            print(folder)
            for files in os.listdir(path+'/'+folder):
                if files[-7:] == '.pickle':
                    data = read_data(path+'/'+folder+'/'+files, True)
                    r.append(data)
    res = pd.concat(r)
    return res


def write_data(path=PATH, filename=FILENAME):
    data = concat_data(path)
    data.to_pickle(path+'/'+filename)


def lifetime_of_data(data):
    return data[data.columns[16401:]]


def intensity_of_data(data):
    return data[data.columns[17:16401]]


def extract_label(data):
    return data['tissue_classification']


def extract_feature(data):
    return intensity_of_data(data)


def build_kmeans_model(**kwargs):
    return KMeans(**kwargs)


def build_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


def get_model(x_train, y_train, **kwargs):
    model = build_random_forest_model(**kwargs)
    model.fit(x_train, y_train)
    return model


def run_model(x_train, y_train, test, **kwargs):
    print('Fit model')
    model = get_model(x_train, y_train, **kwargs)
    print('Test model')
    return model.predict(test)


def main(path=PATH, filename=FILENAME, random_set=False):
    if random_set:
        print('Set seed')
        np.random.seed(RANDOM_SEED)
    if filename not in os.listdir(path):
        raise Exception('File not found')
        # write_data(path, filename)
    print('Read data')
    data = read_data(path+'/'+filename, False)
    data = data.sample(frac=1)
    label = extract_label(data).astype(int)
    data = extract_feature(data)
    cross_validate(run_model, data, label, k=10, max_features=16)


if __name__ == '__main__':
    main(# path='../data/20190208',
         # filename='20190208_13_18_07_CR52.pickle',
         random_set=True)
