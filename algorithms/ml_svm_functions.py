from time import time
from datetime import timedelta
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV

def linSVC_with_tol_iter_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = LinearSVC(penalty='l2', dual=False, class_weight='balanced')
    params["max_iter"] = int(params["max_iter"])
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def linSVC_with_tol_iter_search(data, tol_start=0, tol_end=-7, iter_start=0, iter_end=7):
    gt0 = time()
    possible_tolerances = 10. ** np.arange(tol_start, tol_end, step=-1)
    possible_iterations = 10. ** np.arange(iter_start, iter_end, step=1)
    print(possible_tolerances)
    print(possible_iterations)

    p_grid = {
        "tol": possible_tolerances,
        "max_iter": possible_iterations
    }
    classifier = LinearSVC(penalty='l2', dual=False, class_weight='balanced')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def rbfSVC_with_C_gamma_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = SVC(kernel="rbf", cache_size=1000.0,
                     random_state=None, max_iter=-1, class_weight='balanced')
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def rbfSVC_with_C_gamma_search(data, C_start=-3, C_end=8, gamma_start=-5, gamma_end=4):
    gt0 = time()
    C_range = 10. ** np.arange(C_start, C_end)
    gamma_range = 10. ** np.arange(gamma_start, gamma_end)

    p_grid = {
        "C": C_range,
        "gamma": gamma_range
    }
    classifier = SVC(kernel="rbf", cache_size=1000.0,
                     random_state=None, max_iter=-1, class_weight='balanced')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])    
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))
