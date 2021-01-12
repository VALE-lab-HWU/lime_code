from sklearn.metrics import confusion_matrix

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
    print(matrix)
    max_diag = max([sum([matrix[(j, (j+i) % len(matrix))]
                         for j in list(range(len(matrix)))])
                    for i in range(len(matrix))])
    print(100 * max_diag / len(label))
    print(list(max(matrix[:, i]) for i in range(len(matrix))))
    print_matrix(matrix, lb)
