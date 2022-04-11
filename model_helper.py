from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier


# return a kmeans instance from sklearn
# alias function, to avoid rewritting
def build_kmeans_model(**kwargs):
    return KMeans(**kwargs)


# return a nearest neighbors instance from sklearn
# alias function, to avoid rewritting
def build_knn_model(**kwargs):
    return KNeighborsClassifier(**kwargs)


# return a gaussian naive bayes instance from sklearn
# alias function, to avoid rewritting
def build_gaussian_nb_model(**kwargs):
    return GaussianNB(**kwargs)


# return a gaussian naive bayes instance from sklearn
# alias function, to avoid rewritting
def build_gaussian_cla_model(**kwargs):
    return GaussianProcessClassifier(**kwargs)


# return a random forest instance from sklearn
# alias function, to avoid rewritting
def build_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


def build_svc_model(**kwargs):
    return SVC(**kwargs)


def build_mlp_model(**kwargs):
    return MLPClassifier(**kwargs)


# return a trained model on x and y
def get_model(x_train, y_train, model_fn=build_mlp_model, **kwargs):
    model = model_fn(**kwargs)
    model.fit(x_train, y_train)
    return model


# train and test a model
def run_model(x_train, y_train, test, save_model=False, proba=False, **kwargs):
    print('Fit model')
    model = get_model(x_train, y_train, **kwargs)
    print('Test model')
    if proba:
        predict = model.predict_proba(test)
    else:
        predict = model.predict(test)
    if save_model:
        return predict, model
    else:
        return predict


# do dendrogram
def build_dendrogram_model():
    return AgglomerativeClustering(compute_distances=True)


def fit_dendrogram(data):
    model = build_dendrogram_model()
    model.fit(data)
    return model
