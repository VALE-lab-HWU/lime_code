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


type_list = ['lf',
             'it',
             'lf_b1',
             'lf_b2',
             'it_b1',
             'it_b2',
             'lf_b1_b2',
             'it_b1_b2',
             'it_lf_b1',
             'it_lf_b2',
             'it_lf_b1_b2']


type_run = ['test', 'train', 'patient', 'holdout']


def setup_dt():
    fn_dict = {'lf': paper_test.get_set_lf, 'it': paper_test.get_set_it,
               'lf_b1': paper_test.get_set_lf_b1,
               'lf_b2': paper_test.get_set_lf_b2,
               'it_b1': paper_test.get_set_it_b1,
               'it_b2': paper_test.get_set_it_b2,
               'lf_b1_b2': paper_test.get_set_lf_b1_b2,
               'it_b1_b2': paper_test.get_set_it_b1_b2,
               'it_lf_b1': paper_test.get_set_it_lf_b1,
               'it_lf_b2': paper_test.get_set_it_lf_b2,
               'it_lf_b1_b2': paper_test.get_set_it_lf_b1_b2}
    data = dh.read_data_pickle('robo/grid/data.pkl')
    return fn_dict, data


def load_model(arg, path='grid'):
    model = dh.read_data_pickle('robo/'+path+'/output_' + arg + '.pkl')
    X, y, p = model['X'], model['y'], model['p']
    return model, X, y, p


def setup_md(arg, fn_dict, data, path='grid'):
    model, X, y, p = load_model(arg, path)
    X2, y2, p2 = fn_dict[arg](data[2], data[3])
    y3, y4 = y2[p2 == '20190227'], y2[p2 != '20190227']
    unique_l = np.unique(y)[::-1]
    return model, [X, y, p], [X2, y2, p2], [y3, y4], unique_l


def exec_on_model(arg, data, fn_dict, fn_init, fn_exec, res_dt,
                  model, d1, d2, d3, unique_l,
                  names=['mlp', 'rf', 'svc', 'knn']):
    res_md = fn_init()
    for i in names:
        print('Model:', i)
        fn_exec(res_md, res_dt, model[i], *d1, *d2, *d3,
                unique_l, arg, i)
    return res_md


def exec_on_data(fn_init, fn_exec, fn_init_md, fn_exec_md,
                 names=['mlp', 'rf', 'svc', 'knn'], path='grid'):
    fn_dict, data = setup_dt()
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo/grid') if i.startswith('output_')]
    res = fn_init()
    for i in list_arg:
        print('Data:', i)
        model, d1, d2, d3, unique_l = setup_md(i, fn_dict, data, path)
        res_md = exec_on_model(i, data, fn_dict, fn_init_md, fn_exec_md, res,
                               model, d1, d2, d3, unique_l, names)
        fn_exec(res, res_md, i, d1, d2, d3, unique_l)
    return res


def fn_main_init():
    pass


def fn_main(res, _, model, X, y, p, X2, y2, *arg):
    tmp = model.predict(X)
    tmp2 = model.predict(X2)
    print('Train matrix')
    mlh.compare_class(tmp, y)
    print('Test matrix')
    mlh.compare_class(tmp2, y2)


def main(arg):
    fn_dict, data = setup_dt()
    model, d1, d2, d3, unique_l = setup_md(arg, fn_dict, data)
    exec_on_model(arg, data, fn_dict, fn_main_init, fn_main, [], model,
                  d1, d2, d3, unique_l)


def matrix_to_dict(matrix):
    return {'tp': matrix[0, 0], 'fp': matrix[1, 0],
            'tn': matrix[1, 1], 'fn': matrix[0, 1]}


def fn_main_all_init_md():
    #  tmp_all tmp2_all tmp3_all tmp4_all best_tmp_all
    return [{}, {}, {}, {}, {}]


def fn_main_all_md(res, res_dt, model, X, y, p, X2, y2, p2, y3, y4,
                   unique_l, arg, md):
    res1 = model.predict(X)
    res2 = model.predict(X2)
    res3, res4 = res2[p2 == '20190227'], res2[p2 != '20190227']
    print('Train matrix')
    mlh.compare_class(res1, y)
    print('Test matrix')
    mlh.compare_class(res2, y2)
    matrix = metrics.confusion_matrix(y, res1, labels=unique_l)
    matrix2 = metrics.confusion_matrix(y2, res2, labels=unique_l)
    matrix3 = metrics.confusion_matrix(y3, res3, labels=unique_l)
    matrix4 = metrics.confusion_matrix(y4, res4, labels=unique_l)
    score = {'name': arg+' '+md+' train',
             **mlh.get_score_verbose_2(res1, y, matrix),
             **matrix_to_dict(matrix)}
    score2 = {'name':  arg+' '+md+' test',
              **mlh.get_score_verbose_2(res2, y2, matrix2),
              **matrix_to_dict(matrix2)}
    score3 = {'name':  arg+' '+md+' test patient',
              **mlh.get_score_verbose_2(res3, y3, matrix3),
              **matrix_to_dict(matrix3)}
    score4 = {'name':  arg+' '+md+' test holdout',
              **mlh.get_score_verbose_2(res4, y4, matrix4),
              **matrix_to_dict(matrix4)}
    res[4][md] = model[1].best_score_
    res[0][md] = score['acc']
    res[1][md] = score2['acc']
    res[2][md] = score3['acc']
    res[3][md] = score4['acc']
    res_dt[0] = res_dt[0].append(score, ignore_index=True)
    res_dt[0] = res_dt[0].append(score2, ignore_index=True)
    res_dt[0] = res_dt[0].append(score3, ignore_index=True)
    res_dt[0] = res_dt[0].append(score4, ignore_index=True)


def fn_main_all_init_dt():
    #  df score_all score2_all score3_all score4_all best_score_all
    return [pd.DataFrame(), {}, {}, {}, {}, {}]


def fn_main_all_dt(res_dt, res_md, arg, *args):
    res_dt[1][arg] = res_md[0]
    res_dt[2][arg] = res_md[1]
    res_dt[3][arg] = res_md[2]
    res_dt[4][arg] = res_md[3]
    res_dt[5][arg] = res_md[4]


def main_all():
    res = exec_on_data(fn_main_all_init_dt, fn_main_all_dt,
                       fn_main_all_init_md, fn_main_all_md)
    res[0].to_csv('robo/grid/res.csv')
    scores = {'train': res[1], 'test': res[2], 'patient': res[3],
              'holdout': res[4], 'best_cv': res[5]}
    pd.to_pickle(scores, 'robo/grid/scores.pkl')


def fn_main_ens_init_md():
    #  res1, res2, res3, res4
    return [[], [], [], []]


def fn_main_ens_md(res, res_dt, model, X, y, p, X2, y2, p2, y3, y4,
                   unique_l, arg, md):
    res[0].append(model.predict(X))
    r2 = model.predict(X2)
    res[1].append(r2)
    res[2].append(r2[p2 == '20190227'])
    res[3].append(r2[p2 != '20190227'])


def fn_main_ens_init_dt():
    return [pd.DataFrame()]


def fn_main_ens_dt(res_dt, res_md, arg, d1, d2, d3, unique_l):
    res = np.average(res_md[0], axis=0) >= 0.25
    res2 = np.average(res_md[1], axis=0) >= 0.25
    res3 = np.average(res_md[2], axis=0) >= 0.25
    res4 = np.average(res_md[3], axis=0) >= 0.25
    matrix = metrics.confusion_matrix(d1[1], res, labels=unique_l)
    matrix2 = metrics.confusion_matrix(d2[1], res2, labels=unique_l)
    matrix3 = metrics.confusion_matrix(d3[0], res3, labels=unique_l)
    matrix4 = metrics.confusion_matrix(d3[1], res4, labels=unique_l)
    score = {'name': arg + ' train',
             **mlh.get_score_verbose_2(res, d1[1], matrix),
             **matrix_to_dict(matrix)}
    score2 = {'name': arg + ' test',
              **mlh.get_score_verbose_2(res2, d2[1], matrix2),
              **matrix_to_dict(matrix2)}
    score3 = {'name': arg + ' test patient',
              **mlh.get_score_verbose_2(res3, d3[0], matrix3),
              **matrix_to_dict(matrix3)}
    score4 = {'name': arg + ' test holdout',
              **mlh.get_score_verbose_2(res4, d3[1], matrix4),
              **matrix_to_dict(matrix4)}
    res_dt[0] = res_dt[0].append(score, ignore_index=True)
    res_dt[0] = res_dt[0].append(score2, ignore_index=True)
    res_dt[0] = res_dt[0].append(score3, ignore_index=True)
    res_dt[0] = res_dt[0].append(score4, ignore_index=True)


def main_ens():
    prefix = 'mrsk_1'
    res = exec_on_data(fn_main_ens_init_dt, fn_main_ens_dt,
                       fn_main_ens_init_md, fn_main_ens_md,
                       names=['mlp', 'rf', 'svc', 'knn'])
    res[0].to_csv('robo/grid/ens_'+prefix+'.csv')


def print_color_res(info, value):
    print(info, end=' ')
    if value < 0.5:
        print(Bcolors.RED, end='')
    elif value > 0.9:
        print(Bcolors.GREEN, end='')
    elif value > 0.70:
        print(Bcolors.YELLOW, end='')
    print(value, Bcolors.NC)


def main_best(arg):
    scores = pd.read_pickle('robo/grid/scores.pkl')
    for i in scores:
        print('-----')
        print(i)
        for j in scores[i]:
            if arg == 'avg':
                s = sum(scores[i][j].values())/4
                s2 = ''
            elif arg == 'max':
                s = max(scores[i][j].values())
                s2 = ' -- ' + list(scores[i][j].keys())[
                    np.argmax(list(scores[i][j].values()))]
            elif arg == 'min':
                s = min(scores[i][j].values())
                s2 = ' -- ' + list(scores[i][j].keys())[
                    np.argmin(list(scores[i][j].values()))]
            print_color_res(j+s2, s)


def main_set_graph(ax, length, title):
    ax.set_xlim(-0.05, length)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title(title)


def main_graph_scores():
    scores = dh.read_data_pickle('robo/grid/scores.pkl')
    dt_list = list(scores['test'].keys())
    mdl_list = list(scores['test'][dt_list[0]].keys())
    for k in scores.keys():
        fig, ax = plt.subplots()
        for i in mdl_list:
            tmp = [scores[k][j][i] for j in dt_list]
            ax.plot(dt_list, tmp, marker='o', label=i)
        main_set_graph(ax, len(dt_list)-0.95, k)
        fig.savefig('robo/grid/graph/'+k+'.png')


def main_graph_ensemble(args):
    ensembles = pd.read_csv('robo/grid/ens_'+args.prefix+'.csv')
    fig, ax = plt.subplots()
    for i in type_run:
        ens_test = ensembles[ensembles['name'].str.contains(i+'$', regex=True)]
        ax.plot(ens_test.name.str.split(' ').str[0], ens_test[args.metric],
                label=i, marker='o')
    main_set_graph(ax, len(ens_test)-0.95, 'Ensemble')
    fig.savefig('robo/grid/graph/ensemble_'+args.prefix+'_'+args.metric+'.png')


# get the best param
# need to factor them by accuracy
def main_param():
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo/grid') if i.startswith('output_')]
    best_params = {}
    for i in list_arg:
        tmp, _, _, _ = load_model(i)
        for j in ['mlp', 'rf', 'svc', 'knn']:
            if j not in best_params:
                best_params[j] = {
                    o: [] for o in tmp[j][1].best_params_.keys()}
            for k in tmp[j][1].best_params_:
                best_params[j][k].append(tmp[j][1].best_params_[k])
    for i in best_params:
        print(i)
        for j in best_params[i]:
            print(j+':', np.unique(best_params[i][j], return_counts=True,
                                   axis=0))


def fn_main_predict_init_md():
    return [{}]


def fn_main_predict_md(res, res_dt, model, X, y, p, X2, y2, p2, y3, y4,
                       unique_l, arg, md):
    res1 = model.predict(X)
    res2 = model.predict(X2)
    res3, res4 = res2[p2 == '20190227'], res2[p2 != '20190227']
    res[0][md] = {}
    res[0][md]['train'] = {'x': res1, 'y': y}
    res[0][md]['test'] = {'x': res2, 'y': y2}
    res[0][md]['patient'] = {'x': res3, 'y': y3}
    res[0][md]['holdout'] = {'x': res4, 'y': y4}


def fn_main_predict_init_dt():
    return [{}]


def fn_main_predict_dt(res_dt, res_md, arg, *args):
    res_dt[0][arg] = res_md


def main_predict():
    res = exec_on_data(fn_main_predict_init_dt, fn_main_predict_dt,
                       fn_main_predict_init_md, fn_main_predict_md)
    pd.to_pickle(res, 'robo/grid/predict.pkl')


def main_read_2_all(args):
    try:
        metric = args.metric
    except NameError:
        metric = 'acc'
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo/best_out')
                if i.startswith('output_')]
    res_pt = []
    for i in list_arg:
        data = dh.read_data_pickle('robo/best_out/output_'+i+'.pkl')
        fig, ax = plt.subplots()
        res_ds = {}
        for j in range(len(data)):
            d = data[j]
            if len(res_pt) <= j:
                res_pt.append({})
            for mdl in d[0]:
                matrix = metrics.confusion_matrix(d[1], d[0][mdl],
                                                  labels=[1, 0])
                score = {**mlh.get_score_verbose_2(d[0][mdl], d[1], matrix)}
                if mdl not in res_ds:
                    res_ds[mdl] = []
                if mdl not in res_pt[j]:
                    res_pt[j][mdl] = []
                res_ds[mdl].append(score[metric])
                res_pt[j][mdl].append(score[metric])
        for mdl in res_ds:
            ax.plot(res_ds[mdl], marker='o', label=mdl)
        main_set_graph(ax, len(data)-0.95, i)
        fig.savefig('robo/best_out/graph/dataset/'+i+'.png')
    for i in range(len(res_pt)):
        pt = res_pt[i]
        fig, ax = plt.subplots()
        for mdl in pt:
            ax.plot(list_arg, pt[mdl], marker='o', label=mdl)
        main_set_graph(ax, len(list_arg)-0.95, 'patient '+str(i))
        fig.savefig('robo/best_out/graph/patient/patient'+str(i)+'.png')


def main_2_print_fn(data, metric):
    res = {}
    total = 0
    for i in range(len(data)):
        d = data[i]
        print(f'\nPatient {i} ({len(d[1])})')
        total += len(d[1])
        for mdl in d[0]:
            if len(d[0][mdl].shape) == 2:
                tmp = np.argmax(d[0][mdl], axis=1)
            else:
                tmp = d[0][mdl]
            matrix = metrics.confusion_matrix(d[1], tmp,
                                              labels=[1, 0])
            score = mlh.get_score_verbose_2(tmp, d[1], matrix)
            print_color_res(mdl+' '+metric, score[metric])
            if mdl not in res:
                res[mdl] = []
            res[mdl].append(score[metric]*len(d[1]))
    print('\n-----------')
    for i in res:
        print(i, sum(res[i])/total)
    print('-----------')


def main_2_print(arg):
    try:
        metric = args.metric
    except NameError:
        metric = 'acc'
    s = arg.file
    data = dh.read_data_pickle(arg.path+s+'.pkl')
    if len(data) != 11:
        for d in data:
            main_2_print_fn(d, metric)
    else:
        main_2_print_fn(data, metric)


def main_2_matrix_fn(data):
    res = {}
    total = 0
    for i in range(len(data)):
        d = data[i]
        print(f'\nPatient {i} ({len(d[1])})')
        total += len(d[1])
        for mdl in d[0]:
            print(mdl)
            if len(d[0][mdl].shape) == 2:
                tmp = np.argmax(d[0][mdl], axis=1)
            else:
                tmp = d[0][mdl]
            mlh.compare_class(tmp, d[1], verbose=2, unique_l=[1, 0])
            if mdl not in res:
                res[mdl] = []


def main_2_matrix(arg):
    s = arg.file
    data = dh.read_data_pickle(arg.path+s+'.pkl')
    if len(data) != 11:
        for d in data:
            main_2_matrix_fn(d)
    else:
        main_2_matrix_fn(data)


if __name__ == '__main__':
    args = parse_args_read()
    if args.input == 'all':
        # compute all result for train, test, patient, holdout
        # store it with all measure in res.csv
        # store in scores.pkl the accuracy for each model for each data
        main_all()
    elif args.input == 'plot':
        main_graph_ensemble(args)
    elif args.input in ['avg', 'max', 'min']:
        # print avg, max or min result in color
        # from scores.pkl
        main_best(args.input)
    elif args.input == 'print':
        main_2_print(args)
    elif args.input == 'matrix':
        main_2_matrix(args)
    elif args.input == 'ens':
        # run ensmble model on data test
        # save result for combination models
        main_ens()
    elif args.input == 'predict':
        main_predict()
    else:
        # compare class and print matrix
        main(args.input)
