from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import AgglomerativeClustering


# return a kmeans instance from sklearn
# alias function, to avoid rewritting
def build_kmeans_model(**kwargs):
    return KMeans(**kwargs)


# return a random forest instance from sklearn
# alias function, to avoid rewritting
def build_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


def build_mlp_model(**kwargs):
    return MLPClassifier(**kwargs)


# return a trained model on x and y
def get_model(x_train, y_train, model_fn=build_mlp_model, **kwargs):
    model = model_fn(**kwargs)
    model.fit(x_train, y_train)
    return model


# train and test a model
def run_model(x_train, y_train, test, **kwargs):
    print('Fit model')
    model = get_model(x_train, y_train, **kwargs)
    print('Test model')
    return model.predict(test)


# do dendrogram
def build_dendrogram_model():
    return AgglomerativeClustering(compute_distances=True)


def fit_dendrogram(data):
    model = build_dendrogram_model()
    model.fit(data)
    return model
