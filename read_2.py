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


def get_acc(score):
    return [score['acc'], score['tpr']]


def get_fn_model(scores, idx_best=0, fn=np.argmax, axis=0, idx=True):
    tmp = np.array(list(scores.values()))
    res = fn(tmp[:, idx_best], axis=axis)
    if idx:
        return tmp[res, :]
    else:
        return res


def get_best(scores):
    return get_fn_model(scores)


def get_min(scores):
    return get_fn_model(scores, fn=np.argmin)


def get_average(scores):
    return get_fn_model(scores, fn=np.average, idx=False,
                        idx_best=range(len(next(iter(scores.values())))))


def for_all_model(d, fn_metrics, fn_model):
    res = {}
    for mdl in d[0]:
        matrix = metrics.confusion_matrix(d[1], d[0][mdl],
                                          labels=[1, 0])
        score = mlh.get_score_verbose_2(d[0][mdl], d[1], matrix)
        res[mdl] = fn_metrics(score)
    return fn_model(res)


def for_all_patient(data, fn_metrics, fn_model, fn_patient):
    res = {}
    for i in range(len(data)):
        print('\nPatient', i)
        d = data[i]
        res[i] = for_all_model(d, fn_metrics, fn_model)
    return fn_patient(res)


if __name__ == '__main__':
    args = parse_args_read()
    if args.input == 'all':
        pass
