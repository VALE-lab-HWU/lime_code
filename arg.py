import argparse


def parse_args():
    argp = argparse.ArgumentParser('arg cv code')
    argp.add_argument('-o', dest='name', type=str,
                      help='Output file', default='res.pkl')
    argp.add_argument('-s', dest='set', type=str, default='lf',
                      help='the dataset to use')
    argp.add_argument('-l', dest='log', type=str, default='out',
                      help='log file name')
    return argp.parse_args()


def parse_args_read():
    argp = argparse.ArgumentParser('arg read result')
    argp.add_argument('-i', dest='input', type=str,
                      help="input info = what to do, or files, it's confusing",
                      default='it')
    argp.add_argument('-p', dest='prefix', type=str,
                      help='the prefix for the file to use for graph',
                      default='mrsk')
    argp.add_argument('-m', dest='metric', type=str,
                      help='the metric to plot', default='acc')
    argp.add_argument('-mode', dest='mode', type=str,
                      help='mode patient, dataset', default='p')
    argp.add_argument('-type', dest='type', type=str,
                      help='type all, avg, best', default='avg')
    return argp.parse_args()


def parse_2():
    argp = argparse.ArgumentParser('arg read result')
    argp.add_argument('-m', dest='metric', type=str,
                      help="the metric to use if relevant",
                      default=['acc'], nargs='*')
    argp.add_argument('-s', dest='set', type=str,
                      help="the dataset to use, if relevant",
                      default=['it'], nargs='*')
    argp.add_argument('-p', dest='patient',
                      help="the patient to use if relevant",
                      default=['avg'], nargs='*')
    argp.add_argument('-md', dest='model', type=str,
                      help="the model to use if relevant",
                      default=['max'], nargs='*')
    argp.add_argument('-x', dest='xaxis', type=str,
                      help='what to use as x axis', default='set',
                      choices=['set', 'metric', 'patient', 'model'])
    argp.add_argument('-e', dest='ensemble', type=str, nargs='*',
                      help='what to use for ensemble', default=['mskr'])
    argp.add_argument('-g', dest='generated', type=str,
                      help='pregenerated or not', default='no')
    argp.add_argument('-c', dest='cross', type=bool,
                      help='cross or not', default=True)
    argp.add_argument('-pr', dest='proba', type=bool,
                      help='proba or not', default=False)
    argp.add_argument('-auc', dest='auc', type=bool, help='auc')
    argp.add_argument('-auct', dest='auct', type=str,
                      help='what type of auc', default='sp',
                      choices=['sp', 'sm', 'ms'])
    return argp.parse_args()
