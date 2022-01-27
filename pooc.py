import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

FILEPATH = '../data/processed/cleaned/band_cleaned_MDCEBL_all_patient.pickle'


def main(filepath=FILEPATH):
    data = pd.read_pickle(filepath)
    intensity = data[data.columns[17:16401]]
    label = data['tissue_classification'].astype(int)
    pca = PCA(n_components=0.95, svd_solver='full')
    model = RandomForestClassifier()
    processed_it = pca.fit_transform(intensity)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_it, label, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print('acc:', sum((predicted == y_test)) / len(y_test))


if __name__ == '__main__':
    main()
