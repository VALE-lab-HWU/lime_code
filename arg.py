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
                      help='input file', default='it')
    return argp.parse_args()
