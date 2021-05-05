import numpy as np
from bcolors import Bcolors as bc
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import process_helper as ph
import re
####
# MATRIX PRINTING
####
REX = re.compile('\\033\[\d+m')


# utility function to print a line of a certain length per cell
# lng will be the number of 'cell'
# L will be the width of a cell
def print_line_matrix(lng, L=8):
    print('-' * ((L+1) * (lng) + 1))


# format a string so it fits in a cell
# cutted to L characters (L-1 +'\' actually)
# Numbers have thousands separator if possible
# string are centered.
# It's possible to right aligne numbers but I don't like it
def format_string(ele, L=8):
    ele = str(ele)
    colors = REX.findall(ele)
    value = sorted(REX.split(ele))[-1]
    if value.replace('.', '').isdigit():
        if value.isdigit():
            f_value = int(value)
        else:
            f_value = float(value)
        tmp = '{:,}'.format(f_value).replace(',', ' ')
        if len(tmp) < L:
            value = tmp
    if len(value) > L:
        value = value[:L-1]+'\\'
    value = value[:L].center(L)
    return ''.join(colors[:-1])+value+''.join(colors[-1:])


# function to format the row of a matrix
# r are the different cell
# L is the width for a cell
def format_row(r, L=8):
    return '|' + '|'.join([format_string(i, L) for i in r]) + '|'


# print a 2d array based on a layout
# each cell will have L characters
# can have color code
def print_matrix(layout, L=8):
    print_line_matrix(len(layout[0]), L)
    for i in range(len(layout)):
        print(format_row(layout[i], L))
        len_l = len(layout[i])
        if i + 1 < len(layout):
            len_l = max(len(layout[i+1]), len_l)
        print_line_matrix(len_l, L)



# get multiple values out of a confusion matrix
# recall (tpr)
# precition (ppv)
def get_score_main(matrix):
    res = {}
    res['tpr'] = matrix[0][0] / (matrix[:, 0].sum())
    res['ppv'] = matrix[0][0] / matrix[0].sum()
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
def get_score_predicted(matrix):
    res = {}
    res['tpr'] = matrix[0][0] / matrix[:, 0].sum()
    res['fnr'] = 1 - res['tpr']
    res['fpr'] = matrix[0][1] / matrix[:, 1].sum()
    res['tnr'] = 1 - res['fpr']
    return res


# get multiple values out of a confusion matrix
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
def get_score_label(matrix):
    res = {}
    res['ppv'] = matrix[0][0] / matrix[0].sum()
    res['fdr'] = 1 - res['ppv']
    res['for'] = matrix[1][0] / matrix[1].sum()
    res['npv'] = 1 - res['for']
    return res


# get multiple values out of a confusion matrix
# accuracy (acc)
# prevalence (pre)
def get_score_total(matrix):
    res = {}
    res['acc'] = sum(matrix.diagonal()) / matrix.sum()
    res['pre'] = matrix[:, 0].sum() / matrix.sum()
    return res


# get multiple values out of scores of a classification
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
def get_score_ratio(score):
    res = {}
    res['lr+'] = score['tpr'] / score['fpr'] if score['fpr'] != 0 else float('inf')
    res['lr-'] = score['fnr'] / score['tnr'] if score['tnr'] != 0 else float('inf')
    return res


# get the f1 value  out of scores of a classification
def get_score_f1(score):
    res = {}
    denom = (score['ppv'] + score['tpr'])
    if denom == 0:
        res['f_1'] = 0
    else:
        res['f_1'] = 2.0 *  (score['ppv'] * score['tpr']) / denom
    return res


# get multiple values out of scores of a classification
# f1 score (f_1)
# diagnostic odds ratio (dor)
def get_score_about_score(score):
    res = get_score_f1(score)
    res['dor'] = score['lr+'] / score['lr-']
    return res


# get all values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
# accuracy (acc)
# prevalence (pre)
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
# f1 score (f_1)
# diagnostic odds ratio (dor)
# area under the roc curve (auc)
def get_all_score(predicted, label, matrix):
    res = get_score_predicted(matrix)
    res = {**res, **get_score_label(matrix)}
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_ratio(res)}
    res = {**res, **get_score_about_score(res)}
    res['auc'] = metrics.roc_auc_score(label, predicted)
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# precision (ppv)
# accuracy (acc)
# prevalence (pre)
# f1 score (f_1)
# area under the roc curve (auc)
def get_score_verbose_2(predicted, label, matrix):
    res = get_score_main(matrix)
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_f1(res)}
    res['auc'] = metrics.roc_auc_score(label, predicted)
    return res


####
# Utility
####
blue = ['ppv', 'tpr', 'auc', 'f_1', 'acc']
yellow = ['tnr', 'npv', 'lr+']


# add color to a layout for pretty printing
# color the true in green and false in red
# color the value in the array above (blue, yellow) in blue or yellow
def add_color_layout(layout):
    layout[1][1] = bc.LGREEN + str(layout[1][1]) + bc.NC
    layout[2][2] = bc.LGREEN + str(layout[2][2]) + bc.NC
    layout[1][2] = bc.LRED + str(layout[1][2]) + bc.NC
    layout[2][1] = bc.LRED + str(layout[2][1]) + bc.NC
    # this should be a function somewhere, to much copy paste
    for i in range(0, min(len(layout), 4)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LCYAN + layout[ii][j] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LYELLOW + layout[ii][j] + bc.NC
    for i in range(4, len(layout)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LCYAN + layout[i][jj] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LYELLOW + layout[i][jj] + bc.NC


# append a series of value to the end of the first lines of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each row
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g],[h,i,j],[k,l,m]]
# inv: [0,1]
#
# layout = [[a,b,c,'acc','auc'],
#           [e,f,g,score['acc'],score['auc']],
#           [h,i,j,score['pre']],
#           [k,l,m,'pre']]
def append_layout_col(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele) - len(inv), dtype=int))
    for i in range(len(ele)):
        layout[i*2+(inv[i] % 2)] += ele[i]
        layout[i*2+((1+inv[i]) % 2)] += [score[j] for j in ele[i]]


# append a series of value to the end of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each col
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g]]
# inv: [0,1]
#
# layout = [[a,b,c],
#           [e,f,g],
#           ['acc', score['acc'], score['pre'], 'pre'],
#           [score['auc'],'auc']]
def append_layout_row(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele[0]) - len(inv), dtype=int))
    to_print = [[k for i in range(len(ele[j]))
                 for k in
                 ([ele[j][i], score[ele[j][i]]]
                 if inv[i] == 0 else
                 [score[ele[j][i]], ele[j][i]])]
                for j in range(len(ele))]
    layout.extend(to_print)


# clean a 2d array to make it ready for formatting
# round float and convert all element to string
def clean_layout(layout):
    layout = [[str(round(i, 3)) if isinstance(i, float) else str(i) for i in j]
              for j in layout]
    return layout


# print a comparison of the result of a classification
# label vs predicted
# it will print a confusion matrix
# verbose: how much measure are to be displayed (0,1,2,3)
# color: put color in
# L: cell width
def compare_class(predicted, label, verbose=1, color=True, L=8):
    unique_l = np.unique(label)[::-1]
    matrix = metrics.confusion_matrix(
        label, predicted, labels=unique_l).transpose()
    layout = [['pr\lb', *unique_l],
              [unique_l[0], *matrix[0]],
              [unique_l[1], *matrix[1]]]
    if (verbose > 0):
        layout.append(['total', matrix[:, 0].sum(),
                       matrix[:, 1].sum(), matrix.sum()])
        layout[0].append('total')
        layout[1].append(matrix[0].sum())
        layout[2].append(matrix[1].sum())
        if (verbose == 1):
            score = get_score_total(matrix)
            append_layout_col([['acc'], ['pre']], score, layout)
        elif (verbose == 2):
            score = get_score_verbose_2(predicted, label, matrix)
            append_layout_col([['ppv', 'acc'], ['f_1', 'pre']], score, layout)
            append_layout_row([['tpr', 'auc']], score, layout)
        elif (verbose == 3):
            score = get_all_score(predicted, label, matrix)
            append_layout_col([['ppv', 'fdr', 'acc'],
                               ['for', 'npv', 'pre']],
                              score, layout, inv=[0, 1])
            append_layout_row([['tpr', 'fpr', 'f_1'],
                               ['fnr', 'tnr', 'auc'],
                               ['lr+', 'lr-', 'dor']],
                              score, layout, inv=[0, 1, 0])
    layout = clean_layout(layout)
    if color:
        add_color_layout(layout)
    print_matrix(layout, L)


# get all index where the label and the predicted value are equals to a tuple
def get_index_label_tpl(predicted, label, tpl):
    res = np.nonzero(np.logical_and((predicted == tpl[0]), (label == tpl[1])))
    return res[0]


# get the index of all false positive
def get_index_false_positive(predicted, label):
    return get_index_label_tpl(predicted, label, (1, 0))


# get the index of all false negative
def get_index_false_negative(predicted, label):
    return get_index_label_tpl(predicted, label, (0, 1))


# get the index of all true negative
def get_index_true_negative(predicted, label):
    return get_index_label_tpl(predicted, label, (0, 0))


# get the index of all true positive
def get_index_true_positive(predicted, label):
    return get_index_label_tpl(predicted, label, (1, 1))


# get the index where predicted is different from label
def get_index_mislabeled(predicted, label):
    return np.nonzero(predicted != label)[0]


# get the index where predicted is equal to label
def get_index_well_labeled(predicted, label):
    return np.nonzero(predicted == label)[0]


# return a dictionnary with all indexes of the different classification
# false positive (fp)
# false negative (fn)
# true positive (tp)
# true negative (tn)
# mislabeled (f)
# well labeled (t)
def get_index_claffication(predicted, label, t_f=True):
    res = {}
    res['fp'] = get_index_false_positive(predicted, label)
    res['fn'] = get_index_false_negative(predicted, label)
    res['tp'] = get_index_true_positive(predicted, label)
    res['tn'] = get_index_true_negative(predicted, label)
    if t_f:
        res['f'] = get_index_mislabeled(predicted, label)
        res['t'] = get_index_well_labeled(predicted, label)
    return res


# shuffle multiple arrays in the same order
def shuffle_arrays_of_array(*arrays):
    perm = np.random.permutation(len(arrays[0]))
    return [array[perm] for array in arrays]


####
# Cross Validation
####
# run a cross valisation, using fn as classifier
# datas and label are array of data/label
# each element of data/label will be a fold
def run_cross_validation(fn, datas, labels, **kwargs):
    res = []
    for i in range(len(datas)):
        print('fold %d' % i)
        x_train = np.concatenate(np.concatenate((datas[:i], datas[i+1:])))
        y_train = np.concatenate(np.concatenate((labels[:i],
                                                 labels[i+1:])))
        x_train, y_train = shuffle_arrays_of_array(x_train, y_train)
        x_test = datas[i]
        y_test = labels[i]
        predicted = fn(x_train, y_train, x_test, **kwargs)
        res.append((predicted, y_test))
    return res


# run a k fold cross validation
# fn is the classifierm data and label will be split k times
def cross_validate(fn, data, label, k=10, **kwargs):
    datas = np.array(np.array_split(data, k))
    labels = np.array(np.array_split(label, k))
    return run_cross_validation(fn, datas, labels, **kwargs)


####
# test and validation
####
# run a simple train and test classification
# split the dataset according to percent (percent for train 1-percent for test)
def run_train_and_test(fn, data, label, percent=0.7, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, train_size=percent)
    predicted = fn(x_train, y_train, x_test, **kwargs)
    return (x_test, predicted, y_test)


def get_data_label_from_patient(p_index, patient,  data, label):
    index = np.isin(patient, p_index)
    return data[index], label[index]


#
def get_train_and_test(p_train, p_test, data, label, patient):
    x_train, y_train = get_data_label_from_patient(
        p_train, patient, data, label)
    x_test, y_test = get_data_label_from_patient(p_test, patient, data, label)
    return x_train, x_test, y_train, y_test


# run a simple train and test classification
# split the dataset according by patient.
# a list of patient is given for the training and testing
# the list can be created easily using utility from data_helper
# see the main_one_run_patient_split function in app
def run_train_and_test_patient(
        fn, data, label, patient, p_train, p_test, **kwargs):
    x_train, x_test, y_train, y_test = get_train_and_test(
        p_train, p_test, data, label, patient)
    predicted = fn(x_train, y_train, x_test, **kwargs)
    return (x_test, predicted, y_test)


####
# Pipeline
####
# from lime tutorial
def identity(x):
    return x

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func, inverse_step=identity):
        self._step_func = step_func
        self._inverse_step = inverse_step

    def fit(self, *args):
        return self

    def transform(self, X):
        return self._step_func(X)

    def inverse_transform(self, X):
        return self._inverse_step(X)


# Pipeline for converting color image to gray
# input: rgb 2d
# step1: gray 2d
# step2: gray 1s
def build_pipeline_to_gray(flatten=True):
    makegray_step = PipeStep(ph.gray_imgs)  # , ph.color_imgs)
    if flatten:
        flatten_step = PipeStep(ph.flatten_data)  # , ph.reshape_imgs)
    else:
        flatten_step = PipeStep(identity)
    return Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),
    ])


# Pipeline for converting image to color
# input: 1d float array
# step1: put in float from 0 to 1
# step2: reshape
# step3: color
def build_pipeline_to_color(reshape=True):
    scale_float_step = PipeStep(ph.scale_img_float)
    if reshape:
        reshape_step = PipeStep(ph.reshape_imgs)
    else:
        reshape_step = PipeStep(identity)
    color_step = PipeStep(ph.color_imgs)
    return Pipeline([
        ('Scale to float 0-1', scale_float_step),
        ('Shape 2d image', reshape_step),
        ('Color Img', color_step)
    ])


# pipeline to process images
# step1: scale
# step2: PCA
# step3: scale again
def build_pipeline_pca_scale(
        s1_kwargs={},
        pca_kwargs={'n_components': 0.95, 'svd_solver': 'full'},
        s2_kwargs={}):
    return Pipeline([
        ('Scale data ', StandardScaler(**s1_kwargs)),
        ('PCA', PCA(**pca_kwargs)),
        ('Scale PCA', StandardScaler(**s2_kwargs))
    ])


# pipeline to classify and use lime
# step1: 2d color to 1d gray image
# step2: scale and pca image
# step3: fit
def build_pipeline_classify(model, model_kwargs={}, pipeline_kwargs={}):
    step1 = build_pipeline_to_gray()
    step2 = build_pipeline_pca_scale(**pipeline_kwargs)
    step3 = model(**model_kwargs)
    return Pipeline([
        ('Gray Img ', step1),
        ('Process PCA', step2),
        ('Fit Img', step3)
    ])


# doesn't work
def build_pipline_gray_clasify(model, model_kwargs={}):
    step1 = build_pipeline_to_gray()
    step2 = model(**model_kwargs)
    return Pipeline([
        ('Gray Img ', step1),
        ('Fit Img', step2)
    ])

# doesn't work
def build_pipeline_pca_rgb(pipeline_kwargs={}):
    step1 = build_pipeline_pca_scale(**pipeline_kwargs)
    step2 = PipeStep(ph.pad_arrays_to_square)
    step3 = build_pipeline_to_color(reshape=False)
    step4 = PipeStep(ph.reshape_imgs)
    return Pipeline([
        ('Process PCA', step1),
        ('Pad Square', step2),
        ('Color Img ', step3),
        ('Reshape', step4)
    ])
