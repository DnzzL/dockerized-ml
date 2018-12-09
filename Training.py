import logging

import fire
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def main(input_path, output_path):
    # Import dataset
    logging.info(f"reading {input_path}")
    iris = pd.read_csv(input_path)
    X = iris.drop("Species", axis=1)
    y = iris.Species
    # Instantiate PCA
    pca = PCA()
    # Instatiate logistic_regression
    logistic = SGDClassifier(loss='log', penalty='l2', max_iter=100, tol=1e-3, random_state=0)
    # Parameters grid to try
    param_grid = {
        'pca__n_components': [2, 3],
        'logistic__alpha': np.logspace(-4, 4, 5),
    }
    # Define training pipeline
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    # Training
    logging.info("beginning training")
    search = GridSearchCV(pipe, param_grid, iid=False, cv=3, return_train_score=False)
    search.fit(X, y)
    print(f"Best parameter (CV score={search.best_score_}):")
    print(search.best_params_)
    # Save best model
    logging.info("saving best model")
    dump(search.best_estimator_, output_path)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)
