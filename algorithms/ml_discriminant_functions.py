from time import time
from datetime import timedelta
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV


def LDA_with_tol_iter_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def QDA_with_tol_iter_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = QuadraticDiscriminantAnalysis(store_covariance=True)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block
