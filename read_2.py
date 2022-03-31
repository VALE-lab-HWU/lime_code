import data_helper as dh
from arg import parse_2
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
def reduce_dict(res, a=0, b=0):
    if type(res) == dict:
        if len(res) == 1:
            res = res[list(res.keys())[0]]
            res, a, b = reduce_dict(res)
        else:
            for i in res:
                # print('--', a, i)
                res[i], a, b = reduce_dict(res[i])
            a += 1
        if a == 1:
            b = len(res)
    return res, a, b


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


def apply_dict_level(res, fn, lv=1, **kwargs):
    if type(res) == dict:
        if lv == 0:
            # print('lv', lv)
            tmp = fn(res, **kwargs)
        else:
            tmp = {}
            for m, i in enumerate(res):
                tmpi = [i]
                if 'i' in kwargs:
                    tmpi = [*tmpi, *kwargs['i']]
                tmp[i] = apply_dict_level(res[i], fn, lv-1,
                                          **{**kwargs, 'm': m, 'i': tmpi})
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
def get_fn_model(ivd, fn=np.argmax, axis=0, idx=True, idx_used=None):
    if idx_used is None:
        idx_used = list(ivd.keys())[0]
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


def get_for_model(scores, to_gets, **kwargs):
    res = {}
    keys = list(scores.values())[0].keys()
    # print('k', keys)
    # print('s', scores)
    # print('t', to_gets)
    for to_get in to_gets:
        # print('---vt', to_get)
        # print('res b', res)
        # print(keys, to_get, type(list(keys)[0]), type(to_get))
        if to_get in LST:
            if to_get == 'all':
                res = {**res, **invert_dict_level(scores, lv=1)}
            else:
                res[to_get] = LST[to_get](scores)
        elif to_get in keys:
            for j in scores:
                if to_get not in res:
                    res[to_get] = {}
                res[to_get][j] = scores[j][to_get]
        else:  # ugly ... and?
            sys.exit('wrong model operator')
    # print('res a', res)
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
        # print('Model', mdl)
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
        # print('Patient', i)
        d = data[i]
        res[str(i)] = for_all_model(d, fn_metrics, fn_model)
    return fn_patient(res)


############
# datasets #
############
def get_dataset(scores, to_gets=['avg']):
    tmp = invert_dict_level(scores, lv=3)
    tmp = apply_dict_level(tmp, partial(get_for_model, to_gets=to_gets), lv=2)
    return tmp


###
# exectute patients function for all patients
# and return datasets-level processed  score
###
def for_all_dataset(datas, fn_metrics, fn_model, fn_patient, fn_dataset):
    res = {}
    for i in datas:
        # print('\nDataset', i)
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


def plot_test(res, ax, i, markers, m):
    i.reverse()
    values = np.array(list(res.values()))
    keys = list(res.keys())
    ax.plot(keys, values, marker=markers[m], label=' '.join(str(j) for j in i))


def plot_try():
    res = for_all(fn_dataset=partial(get_dataset, to_gets=['all']),
                  fn_metrics=partial(get_metrics, to_gets=['acc']),
                  fn_model=partial(get_for_model_inv, to_gets=['max']),
                  fn_patient=partial(get_patient, to_gets=['avg']))
    res2, depth, _ = reduce_dict(res)
    res2 = invert_dict_level(res2, lv=0)
    _, _, length = reduce_dict(res2)
    fig, ax = plt.subplots()
    markers = ['o', 'v', 's', 'p', 'x', '8', '*', 'd', 'h', '1', '.', 'X']
    apply_dict_level(
        res2, plot_test, lv=depth-1, ax=ax, markers=markers, i=[], m=0)
    main_set_graph(ax, length-0.95, 'title')


# md : pt : st: met

# model = max 3
# patient = avg 1,3
# metrics = acc, 0
# set = it 02, 3, 3
# 1/2/3 = graph
# last = axis
def plot_main(args):
    res = for_all(fn_dataset=partial(get_dataset, to_gets=args['set']),
                  fn_metrics=partial(get_metrics, to_gets=args['metric']),
                  fn_model=partial(get_for_model_inv, to_gets=args['model']),
                  fn_patient=partial(get_patient, to_gets=args['patient']))
    # print('----')
    # print(args['xaxis'])
    print(res)
    if args['xaxis'] == 'set':
        res = invert_dict_level(res, lv=2)
        res = invert_dict_level(res, lv=3)
        res = invert_dict_level(res, lv=3)
    elif args['xaxis'] == 'model':
        res = invert_dict_level(res, lv=3)
    elif args['xaxis'] == 'patient':
        res = invert_dict_level(res, lv=1)
        res = invert_dict_level(res, lv=3)
    elif args['xaxis'] == 'metric':
        pass
    print(res)
    res2, depth, length = reduce_dict(res)
    print(res2)
    fig, ax = plt.subplots()
    markers = ['o', 'v', 's', 'p', 'x', '8', '*', 'd', 'h', '1', '.', 'X']
    apply_dict_level(
        res2, plot_test, lv=depth-1, ax=ax, markers=markers, i=[], m=0)
    main_set_graph(ax, length-0.95, 'title')
    plt.show()


if __name__ == '__main__':
    args = parse_2()
    if args.generated == 'no':
        plot_main(args.__dict__)
    else:
        fns = {'best_md_pat':  # best model, all, patient axis
               {'metric': ['acc'],
                'set': ['all'],
                'patient': ['all'],
                'model': ['max'],
                'xaxis': 'patient'},
               'best_md_ds':  # best model, all, set  axis
               {'metric': ['acc'],
                'set': ['all'],
                'patient': ['all'],
                'model': ['max'],
                'xaxis': 'set'},
               'pn':  # all about one patient
               {'metric': ['acc'],
                'set': ['all'],
                'patient': args.patient,
                'model': ['all'],
                'xaxis': 'set'},
               'sn':  # all about one set
               {'metric': ['acc'],
                'set': args.set,
                'patient': ['all'],
                'model': ['all'],
                'xaxis': 'patient'},
               'mdn':  # all about one model
               {'metric': ['acc'],
                'set': ['all'],
                'patient': ['all'],
                'model': args.model,
                'xaxis': 'patient'},
               'best':
               {'metric': ['acc', 'tpr', 'ppv', 'pre'],
                'set': ['max'],
                'patient': ['all'],
                'model': ['max'],
                'xaxis': 'patient'}
        }
        plot_main(fns[args.generated])


"""
{'metric': [''],
'set': [''],
'patient': [''],
'model': [''],
'xaxis': ''}
"""
