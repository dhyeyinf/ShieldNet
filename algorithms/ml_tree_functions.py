from time import time
from datetime import timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV
import numpy as np


def DTree_with_maxFeatures_maxDepth_fixed(data, params):
    gt0 = time()
    block = {"params": [], "Y_predictions": [],
             "Y_actuals": [], "runtime": []}
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best')
    classifier = classifier.set_params(**params)
    block["params"] = classifier.get_params(deep=True)
    classifier.fit(data['X_train'], data['Y_train'])

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:

    n_nodes = classifier.tree_.node_count
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    splits_on_features = []
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
            splits_on_features.append(feature[i])
    print()
    print("feature split selects",splits_on_features)

    Y_predictions = classifier.predict(data['X_test'])
    block["classifier"] = classifier
    block["Y_predictions"] = Y_predictions
    block["Y_actuals"] = data['Y_test']
    block["runtime"] = str(timedelta(seconds=time()-gt0))
    return block


def DTree_with_maxFeatures_maxDepth_search(data, max_depth, max_features):
    gt0 = time()
    possible_features = list(range(2, max_features, 1))
    possible_features.extend(['sqrt', 'log2', None])

    p_grid = {
        "max_features": possible_features
    }

    if max_depth is not None:
        p_grid["max_depth"] = range(1, max_depth+1, 1)
    else:
        p_grid["max_depth"] = max_depth

    classifier = DecisionTreeClassifier(criterion='gini', splitter='best')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    gridsearcher = GridSearchCV(estimator=classifier, param_grid=p_grid, cv=cv, scoring=(
        "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1, return_train_score=True)
    gridsearcher.fit(data["X_train"], data["Y_train"])    
    return gridsearcher.cv_results_, str(timedelta(seconds=time()-gt0))
