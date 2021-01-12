import lime
import pandas as pd
import os


PATH = '../data'
FILENAME = 'all_patient.pickle'


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


def run_model():
    pass


def main(path=PATH, filename=FILENAME):
    if filename not in os.listdir(path):
        write_data(path, filename)
    data = read_data(path+'/'+filename)


if __name__ == '__main__':
    main()
