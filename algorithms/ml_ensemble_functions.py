from time import time
from datetime import timedelta
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV


def RForest_with_maxFeatures_maxDepth_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = RandomForestClassifier(criterion='gini', n_jobs=-1)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    print(classifier.n_features_in_)
    print(classifier.feature_importances_)
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def RForest_with_maxFeatures_maxDepth_search(data, max_depth, max_features):
    gt0 = time()
    possible_features = list(range(2, max_features, 5))
    possible_features.extend(['sqrt', 'log2', None])

    p_grid = {
        "max_features": possible_features
    }

    if max_depth is not None:
        p_grid["max_depth"] = range(1, max_depth+1, 5)
    else:
        p_grid["max_depth"] = max_depth

    classifier = RandomForestClassifier(criterion='gini')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def Adaboost_with_nEstimators_rate_fixed(data, params, base_estimator=DecisionTreeClassifier()):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = AdaBoostClassifier(
        base_estimator=base_estimator, algorithm="SAMME.R", random_state=None)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def Adaboost_with_nEstimators_rate_search(data, n_estimator_max=100, learning_rate_max=1.0, base_estimator=DecisionTreeClassifier()):
    gt0 = time()
    p_grid = {
        "n_estimators": [int(i) for i in np.linspace(10, n_estimator_max, round(n_estimator_max/10), dtype=np.int32)],
        "learning_rate": np.linspace(0.1, learning_rate_max, round(learning_rate_max/0.1))
    }
    classifier = AdaBoostClassifier(
        base_estimator=base_estimator, algorithm="SAMME.R", random_state=None)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def Extratrees_with_nEstimators_rate_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [], "Y_actuals": [], "runtime": []}
    classifier = ExtraTreesClassifier(
        random_state=None, n_jobs=-1, class_weight="balanced")
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def Extratrees_with_nEstimators_rate_search(data, n_estimator_max=100):
    gt0 = time()
    p_grid = {
        "n_estimators": [int(i) for i in np.linspace(10, n_estimator_max, round(n_estimator_max/10), dtype=np.int32)]
    }
    classifier = ExtraTreesClassifier(
        random_state=None, n_jobs=-1, class_weight="balanced")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def GradientBoosting_with_nEstimators_rate_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [], "Y_actuals": [], "runtime": []}
    classifier = GradientBoostingClassifier(
        loss='deviance', subsample=0.5, random_state=None)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def GradientBoosting_with_nEstimators_rate_search(data, n_estimator_max=100, learning_rate_max=1.0):
    gt0 = time()
    p_grid = {
        "n_estimators": [int(i) for i in np.linspace(10, n_estimator_max, round(n_estimator_max/10), dtype=np.int32)],
        "learning_rate": np.linspace(0.1, learning_rate_max, round(learning_rate_max/0.1))
    }
    classifier = GradientBoostingClassifier(
        loss='deviance', subsample=0.5, random_state=None)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def XGBoost_with_nEstimators_rate_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [], "Y_actuals": [], "runtime": []}
    classifier = XGBClassifier(n_jobs=-1)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def XGBoost_with_nEstimators_rate_search(data, n_estimator_max=100, learning_rate_max=1.0):
    gt0 = time()
    p_grid = {
        "n_estimators": [int(i) for i in np.linspace(10, n_estimator_max, round(n_estimator_max/10), dtype=np.int32)],
        "learning_rate": np.linspace(0.1, learning_rate_max, round(learning_rate_max/0.1))
    }
    classifier = XGBClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def Bagging_classifier_with_samples_features_fixed(data, params, base_estimator=DecisionTreeClassifier()):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = BaggingClassifier(
        base_estimator=base_estimator, n_jobs=-1)
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def Bagging_classifier_with_samples_features_search(data, min_samples=0.1, max_samples=1.0, min_features=0.1, max_features=1.0, base_estimator=DecisionTreeClassifier()):
    gt0 = time()
    p_grid = {
        "max_samples": np.linspace(min_samples, max_samples, round(max_samples/min_samples)),
        "max_features": np.linspace(min_features, max_features, round(max_features/min_features))
    }
    classifier = BaggingClassifier(base_estimator=base_estimator)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))


def voting_classifier(data, classifiers):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = VotingClassifier(
        estimators=classifiers, voting='soft', n_jobs=-1)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])
    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block
