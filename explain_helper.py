import matplotlib.pyplot as plt
import matplotlib.lines as mpline
import matplotlib
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hierarchy
import ml_helper as mlh


####
# Histogram
####
def build_histogram(data, n_range=None, n_bins=226):
    fig, axs = plt.subplots(1, 1)
    axs.hist(data, bins=n_bins, range=n_range)
    return fig, axs


def save_histogram(data, title, n_range=None, n_bins=226):
    fig, axs = build_histogram(data.reshape(-1), n_range, n_bins)
    plt.savefig(title + ".png")
    plt.clf()


def plot_histogram(data, title, n_range=None, n_bins=226):
    fig, axs = build_histogram(data.reshape(-1), n_range)
    plt.show(block=False)


# not really used
def save_histogram_per_classification(
        data, index_cl, title, n_range=None, n_bins=226):
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


def save_all_histogram_all_data(data, data_cl, title_):
    title = title_ + 'all'
    n_range = None
    n_bins = 226
    save_histogram(data, title+'_full', n_range, n_bins)
    save_histogram_per_data(data_cl, title, n_range, n_bins)
    data = remove_over_represented_data(data)
    title = title_ + 'corrected'
    save_histogram(data, title+'_full', n_range, n_bins)
    data_cl = {i: remove_over_represented_data(data_cl[i]) for i in data_cl}
    save_histogram_per_data(data_cl, title, n_range, n_bins)




####
# metrics
####
def get_measure(data, axis=None):
    res = {}
    res['min'] = data.min(axis=axis)
    res['avg'] = data.mean(axis=axis)
    res['max'] = data.max(axis=axis)
    mod = stats.mode(data, axis=axis)
    res['%-mod'] = ((mod[1]/data.size)*100).reshape(-1)
    res['mad'] = stats.median_abs_deviation(data, axis=axis)
    res['std'] = data.std(axis=axis)
    # res['med'] = np.median(data, axis=1)
    res['10%'], res['25%'], res['med'], res['75%'], res['90%'] = np.percentile(
        data, [10, 25, 50, 75, 90], axis=axis)
    return res


def print_measure(measure):
    layout = [[], [], [], [], [], []]
    mlh.append_layout_col([['10%', '25%', 'med', '75%', '90%'],
                           ['min', 'avg', 'max', 'std'],
                           ['mad', 'v-mod', '%-mod']],
                          measure, layout)
    mlh.print_matrix(layout)


def get_measure_all_cl(data_cl):
    res = {}
    for i in data_cl:
        res[i] = get_measure(data_cl[i], axis=1)
    return res


####
# dendrogram
####
COLORS_LIST = ['red', 'yellow', 'blue', 'orange', 'cyan', 'pink', 'gray',
               'sienna', 'darkviolet', 'magenta', 'deeppink', 'lime',
               'forestgreen', 'gold', 'limegreen', 'dodgerblue', 'sandybrown',
               'silver', 'tan', 'olive']


def get_dict_color(childrens, label, color=None, default='black'):
    if color is None:
        color = COLORS_LIST
    res = {}
    for i, child in enumerate(childrens):
        for j in child:
            j = int(j)
            if j < len(label):
                res[j] = color[label[j]]
            else:
                c = childrens[j-len(label)]
                if res[c[0]] == res[c[1]]:
                    res[j] = res[c[0]]
                else:
                    res[j] = default
    res[(len(label) - 1)*2] = default
    return res


def make_legend(dict_color):
    return [mpline.Line2D([], [], color=dict_color[i], label=i)
            for i in dict_color]


def plot_dendrogram_from_matrix(linkage_matrix, label_to_color,
                                color=COLORS_LIST, default='black', **kwargs):
    if color is None:
        color = COLORS_LIST
    linkage_matrix = np.array(linkage_matrix)
    dict_color_idx = dict(zip(list(dict.fromkeys(label_to_color)), color))
    dict_color_label = get_dict_color(
        linkage_matrix[:, :2], label_to_color, dict_color_idx)
    dendrogram(linkage_matrix, link_color_func=lambda x: dict_color_label[x],
               **kwargs)
    plt.legend(handles=make_legend(dict_color_idx))


# from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def make_linkage_matrix(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    return linkage_matrix


def plot_dendrogram_from_model(model, label_to_color, color=COLORS_LIST,
                               default='black', **kwargs):
    if color is None:
        color = COLORS_LIST
    linkage_mat = make_linkage_matrix(model)
    plot_dendrogram_from_matrix(linkage_mat, label_to_color, color, **kwargs)


def make_label_from_group(label_to_group, groups, label_groups=None):
    if label_groups is None:
        label_groups = list(range(len(groups)))
    return [label_groups[[i in j for j in groups].index(True)]
            for i in label_to_group]


def tmp(dcoords, icoords, color_leaves, color_branch, leaves, ax):
    order = icoords[:, 0].argsort()
    dc0 = dcoords[order]
    ic0 = icoords[order]
    color_branch_0 = color_branch[order]
    i = 0
    for j in range(len(dc0)):
        for k in [0, 3]:
            k2 = (k % 2)+1
            k3 = (k+2) % 4
            link = [[(ic0[j][k], dc0[j][k]),
                     (ic0[j][k2], dc0[j][k2]),
                     ((ic0[j][3] + ic0[j][0])/2, dc0[j][k3])]]
            if dc0[j][k] == 0:
                col = matplotlib.collections.LineCollection(
                    link, colors=(color_leaves[leaves[i]]))
                i += 1
            else:
                col = matplotlib.collections.LineCollection(
                    link, colors=(color_branch_0[j]))
            ax.add_collection(col)
