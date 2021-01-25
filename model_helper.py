from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


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
