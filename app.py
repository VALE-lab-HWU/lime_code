import numpy as np

import matplotlib.pyplot as plt

import lime_helper as lh
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


def main_one_run_patient_split(
        p_train=range(0, 16), p_test=range(16, 20), index=True,
        path=dh.PATH, filename=dh.FILENAME):
    data, label, patient = dh.get_data_complete(path, filename)
    if index:
        p_idx = dh.get_patient_dict(patient)
        p_train = [p_idx[i] for i in p_train]
        p_test = [p_idx[i] for i in p_test]
    x_test, *res = mlh.run_train_and_test_patient(
        mh.run_model, data, label, patient, p_train, p_test)
    mlh.compare_class(*res, verbose=3, color=True)
    index_cl = mlh.get_index_claffication(*res)
    data_cl = eh.get_data_per_classification(data, index_cl)
    eh.save_all_histogram_all_data(x_test, data_cl, 'graph_explain')


def main_cross(path=dh.PATH, filename=dh.FILENAME):
    data, label, patient = dh.get_data_complete(path, filename)
    res = mlh.cross_validate(mh.run_model, data, label, k=10, max_features=16)
    for i, j in res:
        mlh.compare_class(i, j)
    index_cl = mlh.get_index_claffication(*res)
    eh.save_all_histogram_all_data(data, index_cl, 'graph_explain')


def main_one_run(path=dh.PATH, filename=dh.FILENAME):
    data, label, patient = dh.get_data_complete(path, filename)
    data_test, *res = mlh.run_train_and_test(
        mh.run_model, data, label, max_features=16)
    mlh.compare_class(*res, verbose=3, color=True)
    index_cl = mlh.get_index_claffication(*res)
    data_cl = eh.get_data_per_classification(data, index_cl)
    eh.save_all_histogram_all_data(data_test, data_cl, 'graph_explain')


def main_lime_advanced(p_train=range(0, 16), p_test=range(16, 20), index=True,
                       path=dh.PATH, filename=dh.FILENAME):
    data, label, patient = dh.get_data_complete(path, filename, False)
    if index:
        p_idx = dh.get_patient_dict(patient)
        p_train = [p_idx[i] for i in p_train]
        p_test = [p_idx[i] for i in p_test]
    pip_color = mlh.build_pipeline_to_color()
    pip_process = mlh.build_pipeline_classify(
        mh.build_random_forest_model,
        model_kwargs={'n_jobs': 10, 'n_estimators': 100, 'max_features': 32})
    data = pip_color.transform(data)
    x_train, x_test, y_train, y_test = mlh.get_train_and_test(
        p_train, p_test, data, label, patient)
    pip_process.fit(x_train, y_train)
    predicted = pip_process.predict(x_test)
    mlh.compare_class(predicted, y_test, verbose=3, color=True)
    index_cl = mlh.get_index_claffication(predicted, y_test)
    data_cl = eh.get_data_per_classification(data, index_cl)
    eh.save_all_histogram_all_data(x_test, data_cl, 'graph_explain')
    explainer, segmenter = lh.get_explainer()
    for i in data_cl:
        explanation = explainer.explain_instance(
            data_cl[i][0], classifier_fn=pip_process.predict_proba,
            segmentation_fn=segmenter, num_samples=10000, top_labels=2,
            hide_color=0)
        if i[0] == 't':
            lh.visualize_explanation(explanation, 1)
        else:
            lh.visualize_explanation(explanation, 0)
    plt.show()


if __name__ == '__main__':
    print('Set seed')
    np.random.seed(RANDOM_SEED)
    # main_cross(# path='../data/20190208',
    # filename='20190208_13_18_07_CR52.pickle',
    #     random_set=True)
    #main_per_patient()
    #main_one_run()
    #main_lime()
    main_one_run_patient_split()
