import lime
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from helper import matrix_confusion

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
    return lifetime_of_data(data)


def build_kmeans_model():
    return KMeans(n_clusters=2)


def build_random_forest_model():
    return RandomForestClassifier(max_features=16)


def get_model(data, label):
    model = build_random_forest_model()
    model.fit(data, label)
    return model


def compare_class(predicted, label):
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    found = dict(zip(unique_p, counts_p))
    unique_l, counts_l = np.unique(label, return_counts=True)
    label_nb = dict(zip(unique_l, counts_l))
    print('found: ', found)
    print('label: ', label_nb)
    matrix_confusion(label, predicted, unique_l)
    # for j in range(0, len(unique_l)):
    #     predicted = (predicted + 1) % len(unique_l)
    #     matrix_confusion(label, predicted, unique_l)


def run_model(train, test, y_train, y_test):
    print('Fit model')
    model = get_model(train, y_train)
    print('Test model')
    prediction = model.predict(test)
    print('prediction')
    compare_class(prediction, y_test)
    # print('training labels')
    # compare_class(model.labels_, y_train)


def main(path=PATH, filename=FILENAME, random_set=False):
    if random_set:
        print('Set seed')
        np.random.seed(RANDOM_SEED)
    if filename not in os.listdir(path):
        raise Exception('File not found')
        # write_data(path, filename)
    print('Read data')
    data = read_data(path+'/'+filename, True)
    label = extract_label(data).astype(int)
    data = extract_feature(data)
    (x_train, x_test, y_train, y_test) = train_test_split(
        data, label, test_size=0.33)
    run_model(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main(path='../data/20190208',
         filename='20190208_13_18_07_CR52.pickle',
         random_set=True)
