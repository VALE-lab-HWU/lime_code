import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import ml_helper as mlh


####
# Histogram
####
def build_histogram(data, n_range=None, n_bins=218):
    fig, axs = plt.subplots(1, 1)
    axs.hist(data, bins=n_bins, range=n_range)
    return fig, axs


def save_histogram(data, title, n_range=None, n_bins=218):
    fig, axs = build_histogram(data.reshape(-1), n_range, n_bins)
    plt.savefig(title + ".png")
    plt.clf()


def plot_histogram(data, title, n_range=None, n_bins=218):
    fig, axs = build_histogram(data.reshape(-1), n_range)
    plt.show(block=False)


# not really used
def save_histogram_per_classification(
        data, index_cl, title, n_range=None, n_bins=218):
    save_histogram(data, title + ' full', n_range, n_bins)
    for i in index_cl:
        save_histogram(data[index_cl[i]], title+' '+i, n_range, n_bins)


def remove_over_represented_data(data):
    u, c = np.unique(data, return_counts=True)
    return data[np.isin(data, u[c > (data.size/10)], invert=True)]


def get_data_per_classification(data, index_cl):
    return {i: data[index_cl[i]] for i in index_cl}


def save_histogram_per_data(datas, title, n_range=None, n_bins=218):
    for i in datas:
        save_histogram(datas[i], title+'_'+i, n_range, n_bins)


def save_all_histogram_all_data(data, data_cl, title):
    title = 'all'
    n_range = None
    n_bins = 218
    save_histogram(data, title+'_full', n_range, n_bins)
    save_histogram_per_data(data_cl, title, n_range, n_bins)
    data = remove_over_represented_data(data)
    title = 'corrected'
    save_histogram(data, title+'_full', n_range, n_bins)
    data_cl = {i: remove_over_represented_data(data_cl[i]) for i in data_cl}
    save_histogram_per_data(data_cl, title, n_range, n_bins)

