from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


# return a kmeans instance from sklearn
# alias function, to avoid rewritting
def build_kmeans_model(**kwargs):
    return KMeans(**kwargs)


# return a random forest instance from sklearn
# alias function, to avoid rewritting
def build_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)


# return a trained model on x and y
def get_model(x_train, y_train, **kwargs):
    model = build_random_forest_model(**kwargs)
    model.fit(x_train, y_train)
    return model


# train and test a model
def run_model(x_train, y_train, test, **kwargs):
    print('Fit model')
    model = get_model(x_train, y_train, **kwargs)
    print('Test model')
    return model.predict(test)
