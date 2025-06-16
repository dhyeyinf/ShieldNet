from time import time
from datetime import timedelta
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV


def kNN_with_k_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = KNeighborsClassifier(n_jobs=-1)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def kNN_with_k_search(data, k_min=1, k_max=25, distance_power=2):
    gt0 = time()
    p_grid = {
        "n_neighbors": range(k_min, k_max, 2),
        "p": range(1, distance_power+1)
    }
    classifier = KNeighborsClassifier(n_jobs=-1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def nCentroid_with_metric_threshold_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = NearestCentroid()
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def nCentroid_with_metric_threshold_search(data, metric="euclidean", shrink_threshold=None):
    gt0 = time()
    p_grid = {
        "metric": ["manhattan", "euclidean"],
        "shrink_threshold": np.linspace(0.1, 0.5, 5)
    }
    classifier = NearestCentroid()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])    
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))
