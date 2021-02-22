import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

####
# MATRIX PRINTING
####
L = 8


def print_line_matrix(lng):
    print('-' * ((L+1) * (lng+2) + 1))


def format_string(a):
    return str(a)[:L].center(L)


def format_row(r):
    return '|'.join([format_string(i) for i in r])


def print_matrix(m, lb):
    print_line_matrix(len(lb))
    print('|' + format_string('lb\pr') + '|' + format_row(lb) + '|'
          + format_string('total') + '|')
    print_line_matrix(len(lb))
    for i in range(len(m)):
        print('|' + format_string(lb[i]) + '|' + format_row(m[i]) + '|'
              + format_string(sum(m[i])) + '|')
        print_line_matrix(len(lb))
    print('|' + format_string('total') + '|'
          + format_row(sum(m)) + '|'
          + format_string(m.sum()) + '|')
    print_line_matrix(len(lb))


def get_score_predicted(matrix):
    res = {}
    res['tpr'] = matrix[0][0] / (matrix[1][0] + matrix[0][0])
    res['fnr'] = 1 - res['tpr']
    res['fpr'] = matrix[0][1] / (matrix[1][1] + matrix[0][1])
    res['tnr'] = 1 - res['fpr']
    return res


def get_score_label(matrix):
    res = {}
    res['ppv'] = matrix[0][0] / matrix[:, 0].sum()
    res['fdr'] = 1 - res['ppv']
    res['for'] = matrix[1][0] / matrix[0].sum()
    res['npv'] = 1 - res['for']
    return res


def get_score_total(matrix):
    res = {}
    res['acc'] = sum(matrix.diagonal()) / matrix.sum()
    res['pre'] = matrix[:, 0].sum()
    return res


def get_score_ratio(score):
    res = {}
    res['lr+'] = score['tpr'] / score['fpr']
    res['lr-'] = score['fnr'] / score['tnr']
    return res


def get_score_about_score(score):
    res = {}
    res['dor'] = score['lr+'] / score['lr-']
    res['f_1'] = (score['ppv'] * score['tpr']) / (score['ppv'] + score['tpr'])
    return res


def get_all_score(predicted, label):
    matrix = metrics.confusion_matrix(label, predicted)
    res = get_score_predicted(matrix)
    res = {**res, **get_score_label(matrix)}
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_ratio(res)}
    res = {**res, **get_score_about_score(res)}
    res['auc'] = metrics.roc_auc_score(label, predicted)
    return res


# create and print confusion_matrix
def matrix_confusion(label, predicted, lb):
    matrix = metrics.confusion_matrix(label, predicted)
    # max_diag = max([sum([matrix[(j, (j+i) % len(matrix))]
    #                     for j in list(range(len(matrix)))])
    #                for i in range(len(matrix))])
    # print(100 * max_diag / len(label))
    # print(list(max(matrix[:, i]) for i in range(len(matrix))))
    print(matrix.diagonal().sum() / len(label))
    print_matrix(matrix, lb)


####
# Utility
####
def compare_class(predicted, label):
    unique_p = np.unique(predicted)
    unique_l = np.unique(label)
    matrix_confusion(label, predicted, np.union1d(unique_p, unique_l))
    # for j in range(0, len(unique_l)):
    #     predicted = (predicted + 1) % len(unique_l)
    #     matrix_confusion(label, predicted, unique_l)


def get_index_label_tpl(predicted, label, tpl):
    res = np.nonzero(np.logical_and((predicted == tpl[0]), (label == tpl[1])))
    return res[0]


def get_index_false_positive(predicted, label):
    return get_index_label_tpl(predicted, label, (1, 0))


def get_index_false_negative(predicted, label):
    return get_index_label_tpl(predicted, label, (0, 1))


def get_index_true_negative(predicted, label):
    return get_index_label_tpl(predicted, label, (0, 0))


def get_index_true_positive(predicted, label):
    return get_index_label_tpl(predicted, label, (1, 1))


def get_index_mislabeled(predicted, label):
    return np.nonzero(predicted != label)[0]


def get_index_well_labeled(predicted, label):
    return np.nonzero(predicted == label)[0]


def get_index_claffication(predicted, label):
    res = {}
    res['fp'] = get_index_false_positive(predicted, label)
    res['fn'] = get_index_false_negative(predicted, label)
    res['tp'] = get_index_true_positive(predicted, label)
    res['tn'] = get_index_true_negative(predicted, label)
    res['f'] = get_index_mislabeled(predicted, label)
    res['t'] = get_index_well_labeled(predicted, label)
    return res


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
        compare_class(predicted, y_test)
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
def run_train_and_test(fn, data, label, percent=70, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, train_size=percent)
    predicted = fn(x_train, y_train, x_test, **kwargs)
    compare_class(predicted, y_test)
    return (predicted, y_test)


####
# Pipeline
####
# from lime tutorial
class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func = step_func

    def fit(self, *args):
        return self

    def transform(self, X):
        return self._step_func(X)


# Pipeline for classifying color image
# input: rgb 2d
# step1: gray 2d
# step2: gray 1s
# step3: classifier
def build_pipeline_color(build_model, gray_imgs, flatten_data, **kwargs):
    model = build_model(**kwargs)
    makegray_step = PipeStep(gray_imgs)
    flatten_step = PipeStep(flatten_data)
    return Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),
        ('Classifier', model)
    ])
