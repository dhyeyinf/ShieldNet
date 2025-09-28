import pprint
import json
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
import joblib
pp = pprint.PrettyPrinter(indent=4)

# this method gathers the optimized params from the cross-validation
# it's a voting mechanism, choosing the overall highest rated params
# by looking at its rank in 5 metrics

def extract_optimized_params(cv_results):
    one_locations = []
    possible_params = {}
    for sv in cv_results.keys():
        if re.match('rank_.*', sv):
            one_locations.append(np.ma.getdata(
                cv_results[sv], subok=True).tolist().index(1))
        if re.match('param_.*', sv):
            possible_params[sv] = cv_results[sv]
    majority_vote = np.bincount(one_locations).argmax()
    optimized_params = {}
    for k, v in possible_params.items():
        k_short = re.sub('param_', '', k)
        optimized_params[k_short] = v[majority_vote]
    return optimized_params


def fixed_handler(result, parsed_opts):
    to_serialize = {'estimator': result}
    print(repr(to_serialize['estimator']['classifier']))
    
    to_serialize['accuracy_score'] = accuracy_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['balanced_accuracy'] = balanced_accuracy_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['confusion_matrix'] = confusion_matrix(result['Y_actuals'], result['Y_predictions'])
    to_serialize['f1_score'] = f1_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['precision_score'] = precision_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['recall_score'] = recall_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['roc_auc_score'] = roc_auc_score(result['Y_actuals'], result['Y_predictions'])
    to_serialize['train_percent'] = parsed_opts.trainpercent
    to_serialize['reduced_by'] = parsed_opts.reduceby

    # Pretty print
    print('accuracy_score : ', to_serialize['accuracy_score'])
    print('balanced_accuracy : ', to_serialize['balanced_accuracy'])
    print('confusion_matrix : \n', to_serialize['confusion_matrix'])
    print('f1_score : ', to_serialize['f1_score'])
    print('precision_score : ', to_serialize['precision_score'])
    print('recall_score : ', to_serialize['recall_score'])
    print('roc_auc_score : ', to_serialize['roc_auc_score'])

    # Save model if required
    if parsed_opts.export:
        filename = parsed_opts.resultdir + 'single/models/' + parsed_opts.A + '/' + \
                   f"{parsed_opts.D}_{parsed_opts.A}_{parsed_opts.S}_{parsed_opts.trainpercent}_{parsed_opts.reduceby}.pkl"
        try:
            joblib.dump(result["classifier"], filename, compress=3)
        except KeyError:
            print("Couldn't find the classifier model as a key in the result dict")

    # Save JSON
    if parsed_opts.disk:
        # Clean up and serialize safely
        try:
            to_serialize['estimator']['classifier'] = repr(to_serialize['estimator']['classifier'])
            if 'base_estimator' in to_serialize['estimator']['params']:
                to_serialize['estimator']['params']['base_estimator'] = repr(
                    to_serialize['estimator']['params']['base_estimator']).split('(')[0]
            if 'estimator' in to_serialize['estimator']['params']:
                to_serialize['estimator']['params']['estimator'] = repr(
                    to_serialize['estimator']['params']['estimator']).split('(')[0]
        except KeyError:
            pass

        # Remove numpy arrays and model outputs
        to_serialize['confusion_matrix'] = to_serialize['confusion_matrix'].tolist()
        to_serialize['estimator'].pop('Y_predictions', None)
        to_serialize['estimator'].pop('Y_actuals', None)

        # Ensure output folder exists
        output_dir = os.path.join(parsed_opts.resultdir, 'single', parsed_opts.A)
        os.makedirs(output_dir, exist_ok=True)

        # Write to JSON file
        json_path = os.path.join(output_dir, f"{parsed_opts.D}_{parsed_opts.A}_{parsed_opts.S}_{parsed_opts.trainpercent}_{parsed_opts.reduceby}.json")
        with open(json_path, 'w') as outfile:
            pp.pprint(to_serialize)
            json.dump(to_serialize, outfile)



def search_handler(result, runtime, parsed_opts):
    to_serialize = {k: result[k] for k in result.keys() & {'mean_fit_time', 'mean_score_time', 'mean_test_balanced_accuracy', 'mean_test_f1', 'mean_test_precision',
                                                           'mean_test_recall', 'mean_test_roc_auc', 'mean_train_balanced_accuracy', 'mean_train_f1', 'mean_train_precision', 'mean_train_recall', 'mean_train_roc_auc'}}
    to_serialize['params'] = result['params']
    to_serialize['optimal_params'] = extract_optimized_params(result)
    to_serialize['cv_search_time'] = runtime
    pp.pprint(result)
    print(extract_optimized_params(result))
    if parsed_opts.disk:
        with open(parsed_opts.resultdir+'cv/'+parsed_opts.A+'/'+str(parsed_opts.D)+'_'+parsed_opts.A+'_'+parsed_opts.S+'.json', 'w') as outfile:
            for toli in ('mean_fit_time', 'mean_score_time', 'mean_test_balanced_accuracy', 'mean_test_f1', 'mean_test_precision', 'mean_test_recall', 'mean_test_roc_auc', 'mean_train_balanced_accuracy', 'mean_train_f1', 'mean_train_precision', 'mean_train_recall', 'mean_train_roc_auc'):
                to_serialize[toli] = to_serialize[toli].tolist()
            json.dump(to_serialize, outfile)
