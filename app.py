import numpy as np

import matplotlib.pyplot as plt

from lime_helper import get_explainer, visualize_explanation
import process_helper as ph
import data_helper as dh
import model_helper as mh
import explain_helper as eh
import ml_helper as mlh


RANDOM_SEED = 42


def main_per_patient(path=dh.PATH):
    data, label = dh.get_data_per_files(path)
    res = mlh.run_cross_validation(mh.run_model, data, label, max_features=32)
    for i, j in res:
        mlh.compare_class(i, j)


def main_cross(path=dh.PATH, filename=dh.FILENAME):
    data, label = dh.get_data_complete(path, filename)
    res = mlh.cross_validate(mh.run_model, data, label, k=10, max_features=16)
    for i, j in res:
        mlh.compare_class(i, j)
    index_cl = mlh.get_index_claffication(*res)
    eh.save_all_histogram_all_data(data, index_cl, 'graph_explain')


def main_one_run(path=dh.PATH, filename=dh.FILENAME):
    data, label = dh.get_data_complete(path, filename)
    data_test, *res = mlh.run_train_and_test(
        mh.run_model, data, label, max_features=16)
    mlh.compare_class(*res, verbose=3)
    index_cl = mlh.get_index_claffication(*res)
    eh.save_all_histogram_all_data(data_test, index_cl, 'graph_explain')


def main_lime(path=dh.PATH, filename=dh.FILENAME):
    data, label = dh.get_data_complete(path, filename)
    data, label = ph.take_subset_datas([data, label])
    data = ph.get_color_imgs(data)

    pipl = mlh.build_pipeline_color(mh.build_random_forest_model, ph.gray_imgs,
                                    ph.flatten_data,  max_features=16)
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
    main_one_run()
    #main_lime()
