from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray, label2rgb

from helper import cross_validate, run_cross_validation

PATH = '../data'
FILENAME = 'all_patient.pickle'
RANDOM_SEED = 42


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


def read_data(filepath=PATH+'/'+FILENAME, dict=False):
    data = pd.read_pickle(filepath)
    if dict:
        data = [data[i] for i in data][0]
    return data


def read_all_data(path=PATH):
    listdir = os.listdir(path)
    listdir.sort()
    res = []
    for folder in listdir:
        if folder[0] == '2':
            print(folder)
            for files in os.listdir(path+'/'+folder):
                if files[-7:] == '.pickle':
                    data = read_data(path+'/'+folder+'/'+files, True)
                    res.append(data)
    return res


def concat_data(path=PATH):
    data = read_all_data(path)
    res = pd.concat(data)
    return res


def write_data(path=PATH, filename=FILENAME):
    data = concat_data(path)
    data.to_pickle(path+'/'+filename)


def lifetime_of_data(data):
    return data[data.columns[16401:]]


def intensity_of_data(data):
    return data[data.columns[17:16401]]


def extract_label(data):
    return data['tissue_classification'].to_numpy().astype(int)


def extract_feature(data):
    return intensity_of_data(data).to_numpy()


def build_kmeans_model(**kwargs):
    return KMeans(**kwargs)


def build_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


def get_model(x_train, y_train, **kwargs):
    model = build_random_forest_model(**kwargs)
    model.fit(x_train, y_train)
    return model


def run_model(x_train, y_train, test, **kwargs):
    print('Fit model')
    model = get_model(x_train, y_train, **kwargs)
    print('Test model')
    return model.predict(test)


def get_data_per_files(path):
    data = read_all_data(path=PATH)
    label = np.array([extract_label(d) for d in data])
    data = np.array([extract_feature(d) for d in data])
    return data, label


def main_per_patient(path=PATH):
    data, label = get_data_per_files(path)
    run_cross_validation(run_model, data, label, max_features=32)


def get_data_complete(path=PATH, filename=FILENAME):
    if filename not in os.listdir(path):
        raise Exception('File not found')
        # write_data(path, filename)
    print('Read data')
    data = read_data(path+'/'+filename, False)
    label = extract_label(data)
    data = extract_feature(data)
    return data, label


def main_cross(path=PATH, filename=FILENAME):
    data, label = get_data_complete(path, filename)
    cross_validate(run_model, data, label, k=10, max_features=16)


def scale_data(data, u_bound):
    amax = np.amax(data)
    return data * (u_bound / amax)


def scale_img_int(data):
    return scale_data(data, 255).astype(np.uint8)


def scale_img_float(data):
    return scale_data(data, 1).astype(np.float64)


def transform_data(data, fn):
    return np.array([fn(d) for d in data])


def reshape_imgs(data):
    return transform_data(data, lambda data: np.reshape(data, (128, 128)))


def flatten_imgs(data):
    return transform_data(data, np.ravel)


def color_imgs(data):
    return transform_data(data, gray2rgb)


def gray_imgs(data):
    return transform_data(data, rgb2gray)


def arrays1d_to_color_img(arrays):
    return color_imgs(reshape_imgs(arrays))


def build_pipeline(s1, s2, s3):
    return Pipeline([
        ('Make Gray', s1),
        ('Flatten Image', s2),
        ('RF', s3)
    ])


def cut_datas(datas, size=1000):
    r_array = np.random.randint(datas[0].shape[0], size=size)
    return [data[r_array] for data in datas]


def main_lime(path=PATH, filename=FILENAME):
    data, label = get_data_complete(path, filename)
    data, label = cut_datas([data, label])
    data = scale_img_float(data)
    data = arrays1d_to_color_img(data)

    makegray_step = PipeStep(gray_imgs)
    flatten_step = PipeStep(flatten_imgs)
    model = build_random_forest_model(max_features=16)
    pipl = build_pipeline(makegray_step, flatten_step, model)
    pipl.fit(data, label)

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(
        'quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    explanation = explainer.explain_instance(
        data[0], classifier_fn=pipl.predict_proba, segmentation_fn=segmenter,
        num_samples=10000, top_labels=10, hide_color=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    temp, mask = explanation.get_image_and_mask(
        label[0], positive_only=True, num_features=10, hide_rest=False,
        min_weight=0.01)
    ax1.imshow(label2rgb(mask, temp, bg_label=0, bg_color="white"), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(label[0]))

    temp, mask = explanation.get_image_and_mask(
        label[0], positive_only=False, num_features=10, hide_rest=False,
        min_weight=0.01)
    ax2.imshow(label2rgb(mask, temp, bg_label=0, bg_color="white"), interpolation='nearest')
    ax2.set_title('Positive/Negative Regions for {}'.format(label[0]))

    plt.show()


if __name__ == '__main__':
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    # main_cross(# path='../data/20190208',
    # filename='20190208_13_18_07_CR52.pickle',
    #     random_set=True)
    main_per_patient()
    #main_lime()



# RandomForestClassifier.predict_fn = predict_fn
# model = get_model(data[0], labels[0])
