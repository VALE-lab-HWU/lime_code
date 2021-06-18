from skimage.color import gray2rgb, rgb2gray
import numpy as np
from math import sqrt, ceil


# scale data to have the max being equal to u_bound
def scale_data(data, u_bound):
    return u_bound * (data - np.min(data))/np.ptp(data)


# scale to 255, convert to int
def scale_img_int(data):
    return scale_data(data, 255).astype(np.uint8)


# scale to 1, convert to float
def scale_img_float(data):
    return scale_data(data, 1).astype(np.float64)


####
# data is always a 2+d array
# each array in data will get a function applied to it
####

# apply the function fn to all the element in data
def transform_data(data, fn):
    return np.array([fn(d) for d in data])


# reshape each element in data to a square array
# assume the length is a square
# input: data = 2d array, each row of size X
# output: 3d array, row/row of row are of size sqrt(X)
def reshape_imgs(data):
    dim = int(len(data[0]) ** (1/2))
    return transform_data(data, lambda d: np.reshape(d, (dim, dim, -1)))


# flatten each element of data
def flatten_data(data):
    return transform_data(data, np.ravel)


# input: element of data are 1d or 2d array
# output: element of output are 2d or 3d array
# convert all element to rgb, basically add a dimension, of length 3
def color_imgs(data):
    return transform_data(data, gray2rgb)


# input: element of data are 3d or 4d array
# output: element of output are 2d or 3d array
# convert alfrom importlib import reload  l element to grey. Last dimension should be 3
# 0.2125 R + 0.7154 G + 0.0721 B
def gray_imgs(data):
    return transform_data(data, rgb2gray)


# input: array of 1d array of len X
# output: array of 3d array, square of len sqrt(X), last dimension is 3
def arrays1d_to_color_img(arrays):
    return color_imgs(reshape_imgs(arrays))


# shuffle the datas and return only N element, N = size
# input: datas is an array of data. Each data will be shuffled then cut
def take_subset_datas(datas, size=1000):
    r_array = np.random.randint(datas[0].shape[0], size=size)
    return [data[r_array] for data in datas]


# input: array of 1d array of len X
# output: array of 3d array, square of len sqrt(X), last dimension is 3
# the value are scaled between 0 and 1
def get_color_imgs(data):
    data = scale_img_float(data)
    data = arrays1d_to_color_img(data)
    return data


#
def cut_image(data, s_col=13, e_col=118, s_row=0, e_row=128, size=(128, 128)):
    data = data.reshape(size)
    return data[s_row:e_row, s_col:e_col]


def cut_images(datas, s_col=13, e_col=118, s_row=0, e_row=128,
               size=(128, 128)):
    datas = datas.reshape(-1, *size)
    return datas[:, s_row:e_row, s_col:e_col]


#
def pad_array_to_square(data):
    close_square = ceil(sqrt(data.shape[0]))
    fill_needed = (close_square ** 2) - data.shape[0]
    return np.pad(data, (0, fill_needed))


#
def pad_arrays_to_square(data):
    return transform_data(data, pad_array_to_square)



