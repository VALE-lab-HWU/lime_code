import numpy as np

import matplotlib.pyplot as plt

from ml_helper import cross_validate, run_cross_validation, \
    build_pipeline_color
from lime_helper import get_explainer, visualize_explanation
import process_helper as ph
import data_helper as dh
import model_helper as mh


RANDOM_SEED = 42


def main_per_patient(path=dh.PATH):
    data, label = dh.get_data_per_files(path)
    run_cross_validation(mh.run_model, data, label, max_features=32)


def main_cross(path=dh.PATH, filename=dh.FILENAME):
    data, label = dh.get_data_complete(path, filename)
    cross_validate(mh.run_model, data, label, k=10, max_features=16)


def main_lime(path=dh.PATH, filename=dh.FILENAME):
    data, label = dh.get_data_complete(path, filename)
    data, label = ph.take_subset_datas([data, label])
    data = ph.get_color_imgs(data)

    pipl = build_pipeline_color(mh.build_random_forest_model, ph.gray_imgs, ph.flatten_data,  max_features=16)
    pipl.fit(data, label)

    explainer, segmenter = get_explainer()
    explanation = explainer.explain_instance(
        data[0], classifier_fn=pipl.predict_proba, segmentation_fn=segmenter,
        num_samples=10000, top_labels=10, hide_color=0)

    visualize_explanation(explanation, label[0])
    plt.show()


if __name__ == '__main__':
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    # main_cross(# path='../data/20190208',
    # filename='20190208_13_18_07_CR52.pickle',
    #     random_set=True)
    #main_per_patient()
    main_lime()
