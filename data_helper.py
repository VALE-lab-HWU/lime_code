import pandas as pd
import numpy as np
import os


PATH = '../data'
FILENAME = 'all_patient.pickle'


def read_data(filepath=PATH+'/'+FILENAME, dict=False):
    data = pd.read_pickle(filepath)
    if dict:
        data = [data[i] for i in data][0]
    return data


def read_all_data(path=PATH):
    listdir = os.listdir(path)
    listdir.sort()
    res = []
    for folder in listdir:
        if folder[0] == '2':
            print(folder)
            for files in os.listdir(path+'/'+folder):
                if files[-7:] == '.pickle':
                    data = read_data(path+'/'+folder+'/'+files, True)
                    res.append(data)
    return res


def concat_data(path=PATH):
    data = read_all_data(path)
    res = pd.concat(data)
    return res


def write_data(path=PATH, filename=FILENAME):
    data = concat_data(path)
    data.to_pickle(path+'/'+filename)


def lifetime_of_data(data):
    return data[data.columns[16401:]]


def intensity_of_data(data):
    return data[data.columns[17:16401]]


def extract_label(data):
    return data['tissue_classification'].to_numpy().astype(int)


def extract_feature(data):
    return intensity_of_data(data).to_numpy()


def get_data_per_files(path):
    data = read_all_data(path=PATH)
    label = np.array([extract_label(d) for d in data])
    data = np.array([extract_feature(d) for d in data])
    return data, label


def get_data_complete(path=PATH, filename=FILENAME):
    if filename not in os.listdir(path):
        raise Exception('File not found')
        # write_data(path, filename)
    print('Read data')
    data = read_data(path+'/'+filename, False)
    label = extract_label(data)
    data = extract_feature(data)
    return data, label
