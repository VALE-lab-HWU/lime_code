import matplotlib.pyplot as plt
import matplotlib.lines as mpline
import matplotlib
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hierarchy
import ml_helper as mlh
import model_helper as mh
import lime_helper as lh


N_BINS = 218
X_SIZE = 12.8
Y_SIZE = 7.2


####
# Histogram
####
def build_histogram(data, n_range=None, n_bins=N_BINS):
    fig, axs = plt.subplots(1, 1)
    axs.hist(data, bins=n_bins, range=n_range)
    return fig, axs


def save_histogram(data, title, n_range=None, n_bins=N_BINS,
                   x_size=X_SIZE, y_size=Y_SIZE):
    if n_bins is None:
        if (len(data)) == 0:
            n_bins = 1
        else:
            n_bins = int(data.max() - data.min())
    if n_bins < 10:
        n_bins *= 20
    if n_bins == 0:
        n_bins = 100
    print(n_bins)
    fig, raxs = build_histogram(data.reshape(-1), n_range, int(n_bins))
    fig.set_size_inches(x_size, y_size)
    plt.savefig(title + ".png")
    plt.close('all')


def plot_histogram(data, title, n_range=None, n_bins=N_BINS):
    fig, axs = build_histogram(data.reshape(-1), n_range)
    plt.show(block=False)


def save_histo_and_correct(data, title, n_range=None, n_bins=N_BINS,
                           x_size=X_SIZE, y_size=Y_SIZE):
    save_histogram(data, title, n_range=n_range, n_bins=n_bins,
                   x_size=x_size, y_size=y_size)
    data = remove_over_represented_data(data)
    title = title + '_corrected'
    save_histogram(data, title, n_range=n_range, n_bins=n_bins,
                   x_size=x_size, y_size=y_size)


# not really used
def save_histogram_per_classification(
        data, index_cl, title, n_range=None, n_bins=N_BINS,
        x_size=X_SIZE, y_size=Y_SIZE):
    save_histogram(data, title + '_full', n_range, n_bins)
    for i in index_cl:
        save_histogram(data[index_cl[i]], title+' '+i, n_range, n_bins,
                       x_size=x_size, y_size=y_size)


def remove_over_represented_data(data):
    u, c = np.unique(data, return_counts=True)
    return data[np.isin(data, u[c > (data.size/10)], invert=True)]


def get_data_per_classification(data, index_cl):
    return {i: data[index_cl[i]] for i in index_cl}


def save_histogram_per_data(datas, title, n_range=None, n_bins=N_BINS,
                            x_size=X_SIZE, y_size=Y_SIZE):
    for i in datas:
        save_histogram(datas[i], title+'_'+i, n_range, n_bins,
                       x_size=x_size, y_size=y_size)


def save_all_histogram_all_data(data, data_cl, title_,
                                x_size=X_SIZE, y_size=Y_SIZE):
    title = title_ + '_all'
    n_range = None
    n_bins = None
    save_histogram(data, title+'_full', n_range, n_bins,
                   x_size=x_size, y_size=y_size)
    save_histogram_per_data(data_cl, title, n_range, n_bins,
                            x_size=x_size, y_size=y_size)
    data = remove_over_represented_data(data)
    title = title_ + '_corrected'
    save_histogram(data, title+'_full', n_range, n_bins,
                   x_size=x_size, y_size=y_size)
    data_cl = {i: remove_over_represented_data(data_cl[i]) for i in data_cl}
    save_histogram_per_data(data_cl, title, n_range, n_bins,
                            x_size=x_size, y_size=y_size)


####
# metrics
####
def get_measure(data, axis=None):
    res = {}
    res['eff'] = len(data)
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
        if len(data_cl[i]) > 0:
            res[i] = get_measure(data_cl[i])
    return res


def get_measure_patients(data, patient):
    res = {}
    for p in np.unique(patient):
        tmp = patient == p
        res[p] = get_measure(data[tmp])
    return res


####
# dendrogram
####
COLORS_LIST = ['red', 'yellow', 'blue', 'orange', 'cyan', 'pink', 'gray',
               'sienna', 'darkviolet', 'magenta', 'deeppink', 'lime',
               'forestgreen', 'gold', 'limegreen', 'dodgerblue', 'sandybrown',
               'silver', 'tan', 'olive']


# return a dictionary of color, one color for each link id of a dendrogram
# childrens: childrens_ from a sklearn model or the 2 first columns
# of a linkage matrix from scipy
# label: array to use for coloration
# color: list of color to use
# default. color for ;ulti label link
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


# create a legend for matplotlib for a dendrogram
# use a dictionary (or an array)
# dict_color: {'l1': 'red', 'l2': 'yellow'}
def make_legend(dict_color):
    return [mpline.Line2D([], [], color=dict_color[i], label=i)
            for i in dict_color]


# plot a dendrogram from a linkage matrix from scipy
# and color it using label_to_color
# model: a linkage matrix
# label_to_color: an array of the size of the data fitted in the matrix
# color: the list of color to use
# default: the color for the link with multiple label
# no_plot: don't plot using dendrogram from scipy, instead
#   plot using the homemade function
# n0_label: don't plot the label in the X axis
# kwargs: any argument to give to the dendrogram function from scipy
def plot_dendrogram_from_matrix(linkage_matrix, label_to_color,
                                color=COLORS_LIST, default='black',
                                no_plot=True, no_label=True,
                                no_legend=False, **kwargs):
    if color is None:
        color = COLORS_LIST
    linkage_matrix = np.array(linkage_matrix)
    dict_color_idx = dict(zip(list(dict.fromkeys(label_to_color)), color))
    dict_color_label = get_dict_color(
        linkage_matrix[:, :2], label_to_color, dict_color_idx)
    d_struct = dendrogram(
        linkage_matrix, link_color_func=lambda x: dict_color_label[x],
        no_plot=no_plot, no_labels=no_label, **kwargs)
    if no_plot:
        mh = max(linkage_matrix[:, 2])
        my_plot_dendrogram(
            np.array(d_struct['dcoord']), np.array(d_struct['icoord']),
            dict_color_label, np.array(d_struct['color_list']),
            np.array(d_struct['leaves']), mh, d_struct['ivl'], no_label)
        if not no_legend:
            plt.legend(handles=make_legend(dict_color_idx))


# from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# create a linkage matrix like the one from scipy linkage function
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


# plot a dendrogram from an sklearn model + color it with label_to_color
# model: an AggregativeClustering sklearn object
# label_to_color: an array of the size of the data fitted in the model
# color: the list of color to use
# default: the color for the link with multiple label
# no_plot: don't plot using dendrogram from scipy, instead
#   plot using the homemade function
# n0_label: don't plot the label in the X axis
# kwargs: any argument to give to the dendrogram function from scipy
def plot_dendrogram_from_model(model, label_to_color, color=COLORS_LIST,
                               default='black', no_plot=True, no_label=True,
                               no_legend=False, **kwargs):
    if color is None:
        color = COLORS_LIST
    linkage_mat = make_linkage_matrix(model)
    plot_dendrogram_from_matrix(linkage_mat, label_to_color, color,
                                no_plot=no_plot, no_label=no_label,
                                no_legend=no_legend, **kwargs)


# transform an array of array of index and array of array of value to group
# into an array of value
# the label_groups are the value to replace if specified, otherwise it's int
# ex: label_to_group: [0,1,0,2,1,2,0]
# groups: [[0,1],[2]]
# label_groups: ['train', 'test']
# ['train', 'train', 'train', 'test', 'train', 'test', 'train']
def make_label_from_group(label_to_group, groups, label_groups=None):
    if label_groups is None:
        label_groups = list(range(len(groups)))
    return [label_groups[[i in j for j in groups].index(True)]
            for i in label_to_group]


# transform an array of array of index in array of array of values
# size: number of total elements in indexs
# indexs: array of array of indexs
# label_group: the value to place at each index
def make_label_from_index(indexs, size, label_group=None):
    if label_group is None:
        label_group = list(range(size))
    res = np.zeros(size, dtype=object)
    for i in range(len(indexs)):
        res[indexs[i]] = label_group[i]
    return res


# return an based on a dictionnary index
# each value of the dict is an array of int representing the index
# the returned array has the a value equal to the key for the index
# size is the number of element in total in the values of the dict
# ex: {'tn': [1,2], 'fp': [0]}
# -> ['fp', 'tn', 'tn']
def make_label_from_dict_index(dict_idx, size):
    return make_label_from_index(list(dict_idx.values()), size,
                                 list(dict_idx.keys()))


# return an array
# the array is built using label_to_group and groups
# groups is an array of array of value present in label_to_group
# they will be grouped under the same value
# ex:
#   - label_to_group: [0,1,0,2,1,2,0]
#   - groups: [[0,1],[2]]
#   -> [0, 0, 0, 1, 0, 1, 0]
# then the value equal to the value given in keys_idx/the last array in groups
# will be changed using the dictionnary dict_idx which a dict of index
# ex:
#   - dict_idx: {'p': [0], 'l': [1]}
#   -> [0, 0, 0, 'p', 0, 'l', 0]
# label_group is the label to use during the first part
# if None it will be set to a list of integer, from 0 to length of groups - 1
# keys_idx is the key used to apply change the label in the second part
# if None, it will be set to the last label
def make_label_from_dict_and_group(label_to_group, groups, dict_idx,
                                   label_groups=None, keys_idx=None):
    if label_groups is None:
        label_groups = list(range(len(groups)))
        keys_idx = label_groups[-1]
    elif keys_idx is None:
        keys_idx = label_groups[-1]
    array_groups = np.array(make_label_from_group(label_to_group, groups,
                                                  label_groups),
                            dtype=object)
    gr_dict = make_label_from_dict_index(
        dict_idx, sum(array_groups == keys_idx))
    array_groups[array_groups == keys_idx] = gr_dict
    return array_groups


# set up the label for the dendrogram
# copied from the original function _plot_dendrogram from scipy
def set_up_ax_ticks(ax, ivl):
    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)
    ax.set_xticks(iv_ticks)
    ax.xaxis.set_ticks_position('bottom')
    for line in ax.get_xticklines():
        line.set_visible(False)
    leaf_rot = float(hierarchy._get_tick_rotation(len(ivl)))
    leaf_font = float(hierarchy._get_tick_text_size(len(ivl)))
    ax.set_xticklabels(ivl, rotation=leaf_rot, size=leaf_font)


# homemade rewrite of the _plot_dendrogram function from scipy
# it doesn't have option for orientation
# it takes the different value from the dendrogram function
# color the branch one by one instead of by link
def my_plot_dendrogram(dcoords, icoords, color_leaves, color_branch, leaves,
                       mh, ivl, no_label, ax=None):
    if ax is None:
        ax = plt.gca()
    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = mh + mh * 0.05
    ax.set_ylim([0, dvw])
    ax.set_xlim([0, ivw])
    if not no_label:
        set_up_ax_ticks(ax, ivl)
    idx = (dcoords == 0).all(1)
    ica0 = icoords[idx]
    idc = (ica0[:, 0] + ica0[:, 3]) / 2
    dcoords[:, 2][idx] += 0.5
    dcoords[:, 1][idx] += 0.5
    dcoords[:, 3][np.isin(icoords[:, 3], idc)] += 0.5
    dcoords[:, 0][np.isin(icoords[:, 0], idc)] += 0.5
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
                    link, colors=(color_leaves[leaves[i]]), linewidths=1.3)
                i += 1
            else:
                col = matplotlib.collections.LineCollection(
                    link, colors=(color_branch_0[j]), linewidths=1.3)
            ax.add_collection(col)


# distance
def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)


####
# group
####
def group_dendrogram(x_train, y_train, x_test, y_test, index_cl,
                     patient, p_train, p_test, data, label,
                     folder='./result/'):
    # dendrogram of all dataset
    # dendrogram_full = mh.fit_dendrogram(data)
    with open('./model/hierarchi_not_last_full.pkl', 'rb') as f:
        dendrogram_full = pickle.load(f)
    linkage_mat = make_linkage_matrix(dendrogram_full)
    # color dendrogram label
    plot_dendrogram_from_matrix(linkage_mat, np.concatenate((y_train, y_test)))
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_label.png')
    plt.clf()
    # color dendrogram patient
    plot_dendrogram_from_matrix(linkage_mat, patient)
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_patient.png')
    plt.clf()
    # color dendrogram train test
    plot_dendrogram_from_matrix(linkage_mat,
                                make_label_from_group(
                                    patient, [p_train, p_test],
                                    label_groups=['train', 'test']))
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_train_test.png')
    plt.clf()
    # color dendrogram result classification
    plot_dendrogram_from_matrix(linkage_mat,
                                make_label_from_dict_and_group(
                                    patient, [p_train, p_test], index_cl,
                                    label_groups=['train', 'test']))
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_classification.png')
    plt.clf()
    # dendrogram testing
    dendogram_test = mh.fit_dendrogram(x_test)
    linkage_mat = make_linkage_matrix(dendogram_test)
    # color dendrogram label
    plot_dendrogram_from_matrix(linkage_mat,  y_test)
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_test_label.png')
    plt.clf()
    # color dendrogram result classification
    plot_dendrogram_from_matrix(linkage_mat,
                                make_label_from_dict_index(
                                    index_cl, len(x_test)))
    fig = plt.gcf()
    fig.set_size_inches(12.8, 7.2)
    plt.savefig(folder+'dendrogram_test_classification.png')
    plt.clf()


def group_measure_groups_one(lb, title, data_cl, k=3, folder='./result/'):
    measures = []
    for i in range(len(np.unique(lb))):
        tmp_fn = lb == i
        measure = pd.Series(get_measure(data_cl[title][tmp_fn]),
                            name=f'{title}_{i}')
        measures.append(measure)
    return measures


def group_measure_groups(lb_fn, lb_fp, data_cl, k=3, folder='./result/'):
    m1 = group_measure_groups_one(lb_fn, 'fn', data_cl, k=k, folder=folder)
    m2 = group_measure_groups_one(lb_fp, 'fp', data_cl, k=k, folder=folder)
    return [*m1, *m2]


def group_measure(x_train, x_test, patient, data_cl, data,  folder='./result/',
                  title='measures.csv'):
    measure_all = pd.Series(get_measure(data),
                            name='all')
    measure_train = pd.Series(get_measure(x_train), name='train')
    measure_test = pd.Series(get_measure(x_test), name='test')
    measure_per_patient = pd.DataFrame(get_measure_patients(
        data, patient))
    measure_classification = pd.DataFrame(get_measure_all_cl(data_cl))
    measures = [measure_all, measure_train, measure_test,
                measure_per_patient, measure_classification]
    return measures


def group_histogram(x_train, x_test, data_cl, data, label, folder='/result/'):
    title = folder+'graph_explain'
    save_all_histogram_all_data(x_test, data_cl, title+'_test')
    save_histo_and_correct(x_train, title+'_train')
    save_histo_and_correct(x_test, title+'_test')
    save_histo_and_correct(data[label == 0], title+'_false')
    save_histo_and_correct(data[label == 1], title+'_true')


def group_kmeans_one(data, title,  k=3, folder='./result/'):
    if len(data) > 0:
        mdl = mh.build_kmeans_model(n_clusters=min(k, len(data)))
        mdl.fit(data)
        return mdl.labels_
    else:
        return []


def group_kmeans(data_cl, k=3, folder='./result/'):
    lb_fn = group_kmeans_one(data_cl['fn'], 'fn', k=k, folder=folder)
    lb_fp = group_kmeans_one(data_cl['fp'], 'fp', k=k, folder=folder)
    return lb_fn, lb_fp


def group_lime_one(lb, title, i,  exp, seg, pip_color, data_cl, clf,
                   folder='./result/'):
    tmp = lb == i
    for j in range(min(3, tmp.sum())):
        print(f'j1:{j}')
        explanation = exp.explain_instance(
            pip_color.transform([data_cl[title][tmp][j]])[0],
            classifier_fn=clf, segmentation_fn=seg, num_samples=5000,
            top_labels=2, hide_color=0)
        f = lh.visualize_explanation(explanation, 0)
        f.savefig(folder+f'lime_{title}_lb{i}_n{j}')


def group_lime(lb_fn, lb_fp, data_cl, clf, k=3, folder='./result/'):
    exp, seg = lh.get_explainer()
    pip_color = mlh.build_pipeline_to_color()
    for i in range(k):
        print(f'i:{i}')
        group_lime_one(lb_fn, 'fn', i, exp, seg, pip_color, data_cl, clf,
                       folder=folder)
        group_lime_one(lb_fp, 'fp', i, exp, seg, pip_color, data_cl, clf,
                       folder=folder)


def group(x_train, y_train, x_test, y_test, data_cl, index_cl, patient,
          p_train, p_test, clf,  k=3, folder='./result/'):
    # save comput
    data = np.concatenate((x_train, x_test))
    label = np.concatenate((y_train, y_test))
    group_histogram(x_train, x_test, data_cl, data, label, folder=folder)
    group_dendrogram(x_train, y_train, x_test, y_test, index_cl,
                     patient, p_train, p_test, data, label, folder=folder)
    m = group_measure(x_train, x_test, patient, data_cl, data, folder=folder)
    lb_fn, lb_fp = group_kmeans(data_cl, folder=folder)
    m2 = group_measure_groups(lb_fn, lb_fp, data_cl, k=k, folder=folder)
    measures = pd.concat([*m, *m2], axis=1).transpose()
    measures.to_csv(folder + 'measures.csv')
    group_lime(lb_fn, lb_fp, data_cl, clf, k=k, folder=folder)
