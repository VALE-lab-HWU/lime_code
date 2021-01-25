import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

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


# create and print confusion_matrix
def matrix_confusion(label, predicted, lb):
    matrix = confusion_matrix(label, predicted)
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
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    found = dict(zip(unique_p, counts_p))
    unique_l, counts_l = np.unique(label, return_counts=True)
    label_nb = dict(zip(unique_l, counts_l))
    print('found: ', found)
    print('label: ', label_nb)
    matrix_confusion(label, predicted, np.union1d(unique_p, unique_l))
    # for j in range(0, len(unique_l)):
    #     predicted = (predicted + 1) % len(unique_l)
    #     matrix_confusion(label, predicted, unique_l)


def shuffle_arrays_of_array(*arrays):
    perm = np.random.permutation(len(arrays[0]))
    return [array[perm] for array in arrays]


####
# Cross Validation
####
def run_cross_validation(fn, datas, labels, **kwargs):
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


def cross_validate(fn, data, label, k=10, **kwargs):
    datas = np.array(np.array_split(data, k))
    labels = np.array(np.array_split(label, k))
    run_cross_validation(fn, datas, labels, **kwargs)


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


def build_pipeline_color(build_model, gray_imgs, flatten_data, **kwargs):
    model = build_model(**kwargs)
    makegray_step = PipeStep(gray_imgs)
    flatten_step = PipeStep(flatten_data)
    return Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),
        ('RF', model)
    ])
