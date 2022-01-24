import data_helper as dh
from arg import parse_args_read
import paper_test
import ml_helper as mlh
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from bcolors import Bcolors


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
    data = dh.read_data_pickle('robo/data.pkl')
    return fn_dict, data


def setup_md(arg, fn_dict, data):
    model, X, y, p = load_model(arg)
    X2, y2, p2 = fn_dict[arg](data[2], data[3])
    y3, y4 = y2[p2 == '20190227'], y2[p2 != '20190227']
    unique_l = np.unique(y)[::-1]
    return model, [X, y, p], [X2, y2, p2], [y3, y4], unique_l


def load_model(arg):
    model = dh.read_data_pickle('robo/output_' + arg + '.pkl')
    X, y, p = model['X'], model['y'], model['p']
    return model, X, y, p


def exec_on_model(arg, data, fn_dict, fn_init, fn_exec, res_dt,
                  model, d1, d2, d3, unique_l,
                  names=['mlp', 'rf', 'svc', 'knn']):
    res_md = fn_init()
    for i in names:
        print('Model:', i)
        fn_exec(res_md, res_dt, model[i], *d1, *d2, *d3,
                unique_l, arg, i)
    return res_md


def exec_on_data(fn_init, fn_exec, fn_init_md, fn_exec_md):
    fn_dict, data = setup_dt()
    list_arg = [i.replace('output_', '').replace('.pkl', '')
                for i in os.listdir('robo') if i.startswith('output_')]
    res = fn_init()
    for i in list_arg:
        print('Data:', i)
        model, d1, d2, d3, unique_l = setup_md(i, fn_dict, data)
        res_md = exec_on_model(i, data, fn_dict, fn_init_md, fn_exec_md, res, model, d1, d2, d3, unique_l)
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
    matrix = metrics.confusion_matrix(res1, y, labels=unique_l)
    matrix2 = metrics.confusion_matrix(res2, y2, labels=unique_l)
    matrix3 = metrics.confusion_matrix(res3, y3, labels=unique_l)
    matrix4 = metrics.confusion_matrix(res4, y4, labels=unique_l)
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
    res[0].to_csv('robo/res.csv')
    scores = {'train': res[1], 'test': res[2], 'patient': res[3],
              'holdout': res[4], 'best_cv': res[5]}
    pd.to_pickle(scores, 'robo/scores.pkl')


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
    return pd.DataFrame()


def fn_main_ens_dt(res_dt, res_md, arg, d1, d2, d3, unique_l):
    res = np.average(res_md[0], axis=0) >= 0.5
    res2 = np.average(res_md[1], axis=0) >= 0.5
    res3 = np.average(res_md[2], axis=0) >= 0.5
    res4 = np.average(res_md[3], axis=0) >= 0.5
    matrix = metrics.confusion_matrix(res, d1[1], labels=unique_l)
    matrix2 = metrics.confusion_matrix(res2, d2[1], labels=unique_l)
    matrix3 = metrics.confusion_matrix(res3, d3[0], labels=unique_l)
    matrix4 = metrics.confusion_matrix(res4, d3[1], labels=unique_l)
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
    res_dt = res_dt.append(score, ignore_index=True)
    res_dt = res_dt.append(score2, ignore_index=True)
    res_dt = res_dt.append(score3, ignore_index=True)
    res_dt = res_dt.append(score4, ignore_index=True)


def main_ens():
    res = exec_on_data(fn_main_ens_init_dt, fn_main_ens_dt,
                       fn_main_ens_init_md, fn_main_ens_md)
    res.to_csv('robo/ens3.csv')


def main_best(arg):
    scores = pd.read_pickle('robo/scores.pkl')
    for i in scores:
        print('-----')
        print(i)
        for j in scores[i]:
            if arg == 'avg':
                s = sum(scores[i][j].values())/4
                s2 = ''
            elif arg == 'max':
                s = max(scores[i][j].values())
                s2 = '-- ' + list(scores[i][j].keys())[
                    np.argmax(list(scores[i][j].values()))]
            elif arg == 'min':
                s = min(scores[i][j].values())
                s2 = '-- ' + list(scores[i][j].keys())[
                    np.argmin(list(scores[i][j].values()))]
            print(j, s2, end=' ')
            if s < 0.5:
                print(Bcolors.RED, end='')
            elif s > 0.9:
                print(Bcolors.GREEN, end='')
            elif s > 0.70:
                print(Bcolors.YELLOW, end='')
            print(s, Bcolors.NC)


if __name__ == '__main__':
    args = parse_args_read()
    if args.input == 'all':
        main_all()
    elif args.input in ['avg', 'max', 'min']:
        main_best(args.input)
    elif args.input == 'ens':
        main_ens()
    else:
        main(args.input)
