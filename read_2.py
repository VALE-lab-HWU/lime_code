import data_helper as dh
from arg import parse_args_read
import matplotlib.pyplot as plt
import paper_test
import ml_helper as mlh
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from bcolors import Bcolors


def invert_dict(scores):
    res = {}
    for i in scores:
        for j in scores[i]:
            if j not in res:
                res[j] = {}
            res[j][i] = scores[i][j]
    return res


def get_acc(score):
    return {'acc': score['acc'], 'tpr': score['tpr']}


def get_fn_model(scores, fn=np.argmax, axis=0, idx=True, idx_used='acc'):
    res = {}
    ivd = invert_dict(scores)
    if idx:
        tmp = np.array(list(ivd[idx_used].values()))
        tmp_idx = fn(tmp, axis=axis)
        for i in ivd:
            tmp = np.array(list(ivd[i].values()))
            res[i] = tmp[tmp_idx]
    else:
        for i in ivd:
            tmp = np.array(list(ivd[i].values()))
            res[i] = fn(tmp, axis=axis)
    return res


def get_best(scores):
    return get_fn_model(scores)


def get_min(scores):
    return get_fn_model(scores, fn=np.argmin)


def get_average(scores):
    return get_fn_model(scores, fn=np.average, idx=False)


def for_all_model(d, fn_metrics, fn_model):
    res = {}
    for mdl in d[0]:
        print('Model', mdl)
        matrix = metrics.confusion_matrix(d[1], d[0][mdl],
                                          labels=[1, 0])
        score = mlh.get_score_verbose_2(d[0][mdl], d[1], matrix)
        res[mdl] = fn_metrics(score)
    return fn_model(res)


def best_patient(scores):
    print(scores)
    return scores


def for_all_patient(data, fn_metrics, fn_model, fn_patient):
    res = {}
    for i in range(len(data)):
        print('Patient', i)
        d = data[i]
        res[i] = for_all_model(d, fn_metrics, fn_model)
    return fn_patient(res)


def best_dataset_per_patient(scores):
    scores = invert_dict(scores)
    for i in scores:
        scores[i] = get_best(scores[i])
    return scores


def for_all_dataset(datas, fn_metrics, fn_model, fn_patient, fn_dataset):
    res = {}
    for i in datas:
        print('\nDataset', i)
        data = dh.read_data_pickle('robo/best_out/output_'+i+'.pkl')
        res[i] = for_all_patient(data, fn_metrics, fn_model, fn_patient)
    return fn_dataset(res)


def for_all(**kwargs):
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo/best_out')
                if i.startswith('output_')]
    return for_all_dataset(list_arg, **kwargs)


def main_set_graph(ax, length, title):
    ax.set_xlim(-0.05, length)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title(title)


def plot_best_acc_model():
    res = for_all(fn_dataset=lambda x: x, fn_metrics=get_acc,
                  fn_model=get_best, fn_patient=best_patient)
    fig, ax = plt.subplots()
    for i in res:
        values = np.array(list(res[i].values()))
        keys = list(res[i].keys())
        for j in range(len(next(iter(res[i].values())))):
            ax.plot(keys, values[:, j], marker='o', label=i+' acc')
    main_set_graph(ax, len(values)-0.95, 'best model each')


def plot_avg_acc_per_patient():
    res = for_all(fn_dataset=best_dataset_per_patient, fn_metrics=get_acc,
                  fn_model=get_best, fn_patient=best_patient)
    fig, ax = plt.subplots()
    res = invert_dict(res)
    for i in res:
        values = np.array(list(res[i].values()))
        keys = list(res[i].keys())
        ax.plot(keys, values, marker='o', label=i)
    main_set_graph(ax, len(values)-0.95, 'avg')


if __name__ == '__main__':
    args = parse_args_read()
    if args.input == 'all':
        pass
