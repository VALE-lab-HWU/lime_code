import data_helper as dh
from arg import parse_args_read
import matplotlib.pyplot as plt
import paper_test
import ml_helper as mlh
import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from bcolors import Bcolors
from functools import partial


# general utility
def invert_dict(scores):
    res = {}
    for i in scores:
        for j in scores[i]:
            if j not in res:
                res[j] = {}
            res[j][i] = scores[i][j]
    return res


def invert_dict_level(res, lv=1):
    if type(res) == dict:
        if lv > 0:
            res = invert_dict(res)
        for i in res:
            res[i] = invert_dict_level(res[i], lv-1)
        return res
    else:
        return res


def apply_dict_level(res, fn, lv=1):
    if type(res) == dict:
        if lv == 0:
            print('lv', lv)
            tmp = fn(res)
        else:
            tmp = {}
            for i in res:
                tmp[i] = apply_dict_level(res[i], fn, lv-1)
        return tmp
    else:
        return res


def identity(x):
    return x


###########
# metrics #
###########
def get_acc(score):
    return {'acc': score['acc']}


def get_acc_tpr(score):
    return {'acc': score['acc'], 'tpr': score['tpr']}


def get_metrics(score, to_gets):
    return {i: score[i] for i in to_gets}


##########
# models #
##########
def get_fn_model(ivd, fn=np.argmax, axis=0, idx=True, idx_used='acc'):
    res = {}
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


LST = {'max': get_best, 'min': get_min, 'avg': get_average, 'all': identity}


def get_for_model(scores, to_gets):
    res = {}
    for to_get in to_gets:
        if to_get in LST:
            res[to_get] = LST[to_get](scores)
        else:  # ugly ... and?
            sys.exit('wrong model operator')
    return res


def get_for_model_inv(scores, to_gets):
    scores = invert_dict(scores)
    return get_for_model(scores, to_gets)


###
# exectute metrics on all score
# and return model-level processed score
###
def for_all_model(d, fn_metrics, fn_model):
    res = {}
    for mdl in d[0]:
        print('Model', mdl)
        matrix = metrics.confusion_matrix(d[1], d[0][mdl],
                                          labels=[1, 0])
        score = mlh.get_score_verbose_2(d[0][mdl], d[1], matrix)
        res[mdl] = fn_metrics(score)
    return fn_model(res)


############
# patients #
############
def get_patient(scores, to_gets=['avg']):
    tmp = invert_dict_level(scores, lv=2)
    tmp = apply_dict_level(tmp, partial(get_for_model, to_gets=to_gets), lv=1)
    return tmp


###
# exectute models function for all model
# and return patient-level processed score
###
def for_all_patient(data, fn_metrics, fn_model, fn_patient):
    res = {}
    for i in range(len(data)):
        print('Patient', i)
        d = data[i]
        res[i] = for_all_model(d, fn_metrics, fn_model)
    return fn_patient(res)


############
# datasets #
############
def best_dataset_per_patient(scores):
    scores = invert_dict(scores)
    for i in scores:
        scores[i] = invert_dict(scores[i])
        scores[i] = get_best(scores[i])
    return scores


###
# exectute patients function for all patients
# and return datasets-level processed  score
###
def for_all_dataset(datas, fn_metrics, fn_model, fn_patient, fn_dataset):
    res = {}
    for i in datas:
        print('\nDataset', i)
        data = dh.read_data_pickle('robo/best_out/output_'+i+'.pkl')
        res[i] = for_all_patient(data, fn_metrics, fn_model, fn_patient)
    return fn_dataset(res)


###
# execute for_all_dataset with all data in the robo/best_out folder
###
def for_all(**kwargs):
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo/best_out')
                if i.startswith('output_')]
    return for_all_dataset(list_arg, **kwargs)


###
# utility for graph
# set the axis size, the legend and the title
# y axis is hardcoded as we have proba (0<=x<=1)
###
def main_set_graph(ax, length, title):
    ax.set_xlim(-0.05, length)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title(title)


# def plot_best_acc_model():
#     res = for_all(fn_dataset=lambda x: x, fn_metrics=get_acc,
#                   fn_model=get_best, fn_patient=best_patient)
#     fig, ax = plt.subplots()
#     for i in res:
#         i_res = invert_dict(res[i])
#         for j in i_res:
#             values = list(i_res[j].values())
#             keys = list(i_res[j].keys())
#             ax.plot(keys, values, marker='o', label=i+' '+j)
#     main_set_graph(ax, len(values)-0.95, 'best model each')


def plot_try():
    res = for_all(fn_dataset=identity,
                  fn_metrics=partial(get_metrics, to_gets=['acc']),
                  fn_model=partial(get_for_model_inv, to_gets=['avg']),
                  fn_patient=partial(get_patient, to_gets=['avg']))
    fig, ax = plt.subplots()
    res2 = invert_dict(res)  # patient top
    markers = ['o', 'v', 's', 'p', 'x']
    for m, i in enumerate(res2):
        res2[i] = invert_dict(res2[i])   # model second
        for m2, j in enumerate(res2[i]):
            res2[i][j] = invert_dict(res2[i][j])  # metrics third
            for m3, k in enumerate(res2[i][j]):
                values = np.array(list(res2[i][j][k].values()))
                keys = list(res2[i][j][k].keys())
                ax.plot(keys, values, marker=markers[m3], label=i+' '+j+' '+k)
    main_set_graph(ax, len(values)-0.95, 'title')


    
if __name__ == '__main__':
    args = parse_args_read()
    if args.input == 'all':
        pass


    ## do invert
    ## then apply with level argument
