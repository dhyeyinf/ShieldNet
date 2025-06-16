#!/usr/bin/env -S python -u

import struct
import socket
import sys
import subprocess
from ast import literal_eval
from time import time
from datetime import timedelta
from math import sqrt, log
import json
import pandas as pd
import numpy as np
from color import bcolors as c

from misc import build_parser, remove_ip, split_label_off, train_test_splitter
from result_handling import fixed_handler, search_handler
from preproc import cast_frame, z_scaler, minmax_scaler, df_string_impute
from reduction import tree_important_drop, non_unique_drop, run_pca
from algorithms.ml_neighbor_functions import kNN_with_k_fixed, kNN_with_k_search, nCentroid_with_metric_threshold_fixed
from algorithms.ml_svm_functions import linSVC_with_tol_iter_fixed, linSVC_with_tol_iter_search, rbfSVC_with_C_gamma_fixed
from algorithms.ml_linearmodel_functions import binLR_with_tol_iter_fixed, binLR_with_tol_iter_search
from algorithms.ml_tree_functions import DTree_with_maxFeatures_maxDepth_fixed, DTree_with_maxFeatures_maxDepth_search
from algorithms.ml_ensemble_functions import RForest_with_maxFeatures_maxDepth_fixed, RForest_with_maxFeatures_maxDepth_search, Adaboost_with_nEstimators_rate_fixed, Adaboost_with_nEstimators_rate_search, Bagging_classifier_with_samples_features_search, Bagging_classifier_with_samples_features_fixed, Extratrees_with_nEstimators_rate_fixed, Extratrees_with_nEstimators_rate_search, GradientBoosting_with_nEstimators_rate_fixed, GradientBoosting_with_nEstimators_rate_search, XGBoost_with_nEstimators_rate_search, XGBoost_with_nEstimators_rate_fixed
from algorithms.ml_discriminant_functions import LDA_with_tol_iter_fixed, QDA_with_tol_iter_fixed

# Keep track of time
totaltime = time()

# Gather command line arguments
parsed_opts = build_parser().parse_args()

# Raw data
monday_httpdos = parsed_opts.datadir+"/TestbedMonJun14Flows.csv"
tuesday_ddos = parsed_opts.datadir+"/TestbedTueJun15Flows.csv"
wednesday_bruteforce = parsed_opts.datadir+"/TestbedWedJun16Flows.csv"
thursday_bruteforce_ssh = parsed_opts.datadir+"/TestbedThuJun17Flows.csv"
saturday_bruteforce = parsed_opts.datadir+"/TestbedSatJun12Flows.csv"
sunday_infiltration = parsed_opts.datadir+"/TestbedSunJun13Flows.csv"

days = [monday_httpdos, tuesday_ddos, wednesday_bruteforce,
        thursday_bruteforce_ssh, saturday_bruteforce, sunday_infiltration]

chosen_day = days[parsed_opts.D]

# All columns
col_names = np.array(["generated", "appName", "totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64",
                      "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF", "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription",
                      "source", "protocolName", "sourcePort", "destination", "destinationPort", "startDateTime", "stopDateTime", "Label"]
                     )

# Read the raw data
dataframe = pd.read_csv(chosen_day, names=col_names, skiprows=1)
print(c.OKBLUE, "Reading " + chosen_day + " => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Drop generated col, because it is uninformative
dataframe.drop(axis=1, columns='generated', inplace=True)
col_names = col_names[1::]
print(c.OKBLUE, "Drop flow-id column => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Drop payload columns, because I don't know how to make proper features out of them
dataframe.drop(axis=1, columns=["sourcePayloadAsBase64", "sourcePayloadAsUTF",
                                "destinationPayloadAsBase64", "destinationPayloadAsUTF"], inplace=True)
col_names = np.setdiff1d(col_names, np.array(
    ["sourcePayloadAsBase64", "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF"]))
print(c.OKBLUE, "Drop payload columns => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Source ip -> numeric
dataframe["source"] = dataframe["source"].apply(
    lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
)

# Destination ip -> numeric
dataframe["destination"] = dataframe["destination"].apply(
    lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
)
print(c.OKBLUE, "Translate IPs => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))


def process_time(time):
    time = time.split(" ")[-1]
    h, m = time.split(":")
    return 3600*int(h)+60*int(m)


dataframe["startDateTime"] = dataframe["startDateTime"].apply(
    lambda time: process_time(time)
)

dataframe["stopDateTime"] = dataframe["stopDateTime"].apply(
    lambda time: process_time(time)
)
print(c.OKBLUE, "Process start / stop datetime => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))


def flags_transform(flags):
    value = 0
    if type(flags) is str:
        flags = flags.replace(" ", "")
        for c in flags:
            if c != ',':
                value += ord(c)
    return value


dataframe["sourceTCPFlagsDescription"] = dataframe["sourceTCPFlagsDescription"].apply(
    lambda flags: flags_transform(flags)
)

dataframe["destinationTCPFlagsDescription"] = dataframe["destinationTCPFlagsDescription"].apply(
    lambda flags: flags_transform(flags)
)

# Neat printing
# with pd.option_context('display.max_rows', 1, 'display.max_columns', None):
#     print(dataframe)

# Convert categorical
dataframe["protocolName"] = dataframe["protocolName"].astype("category")
dataframe["protocolName"] = dataframe["protocolName"].cat.codes
dataframe["appName"] = dataframe["appName"].astype("category")
dataframe["appName"] = dataframe["appName"].cat.codes
dataframe["direction"] = dataframe["direction"].astype("category")
dataframe["direction"] = dataframe["direction"].cat.codes

# Remove tree important features
if parsed_opts.A in ('dtree', 'bag', 'ada', 'rforest', 'binlr', 'linsvc', 'rbfsvc', 'knn', 'ncentroid', 'gradboost', 'extratree', 'xgboost'):
    dataframe = tree_important_drop(dataframe, parsed_opts.reduceby)
    print(c.OKBLUE, 'Removed tree features => ', 'Total time elapsed',
          c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Translate string attack types to numbers
attack_dict = {"Normal": 0, "Attack": 1}
dataframe["Label"] = dataframe["Label"].apply(
    lambda x: attack_dict[x]
)
print(c.OKBLUE, "Binarize label => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Print the distribution of the labels
print(c.OKBLUE, "Label distribution:", c.ENDC)
print(dataframe['Label'].value_counts())

# Fixing the NaN values
dataframe = dataframe.fillna(0)
print(c.OKBLUE, "Fill N/A => ", "Total time elapsed",
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Drop columns with only 1 unique value
if parsed_opts.R:
    dataframe = non_unique_drop(dataframe)
    print(c.OKBLUE, 'Dropped columns with only 1 unique value => ', 'Total time elapsed',
          c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Dataframe casting
dataframe = cast_frame(dataframe)
print(c.OKBLUE, 'Cast to float64 => ', 'Total time elapsed',
      c.ENDC, str(timedelta(seconds=time()-totaltime)))

# Preprocessing -> scaling
if parsed_opts.S == 'Z':
    print(c.OKBLUE, 'Standardizing scaler', c.ENDC)
    dataframe = z_scaler(dataframe, dataframe.columns)
    print(c.OKBLUE, 'Z scaling sklearn => ', c.ENDC,
          'Total time elapsed', str(timedelta(seconds=time()-totaltime)))

elif parsed_opts.S == 'MinMax':
    print(c.OKBLUE, 'MinMax scaler', c.ENDC)
    dataframe = minmax_scaler(dataframe, dataframe.columns)
    print(c.OKBLUE, 'MinMax scaling sklearn => ', c.ENDC,
          'Total time elapsed', str(timedelta(seconds=time()-totaltime)))

elif parsed_opts.S == 'No':
    print(c.OKBLUE, "Not applying any scaling", c.ENDC)

if parsed_opts.R == 'pca':
    if parsed_opts.S != 'No':
        dataframe = run_pca(dataframe)
    else:
        sys.exit("Don't use PCA without scaling!")


data_without_label = split_label_off(dataframe)

if parsed_opts.O:
    data_parts_without_label = train_test_splitter(data_without_label, 0.50)
    if parsed_opts.A == 'knn':
        result, cvtime = kNN_with_k_search(
            data_parts_without_label, k_min=1, k_max=5)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'dtree':
        result, cvtime = DTree_with_maxFeatures_maxDepth_search(
            data_parts_without_label, max_depth=35, max_features=data_without_label['X'].shape[1])
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'rforest':
        result, cvtime = RForest_with_maxFeatures_maxDepth_search(
            data_parts_without_label, max_depth=35, max_features=data_without_label['X'].shape[1])
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'ada':
        result, cvtime = Adaboost_with_nEstimators_rate_search(
            data_parts_without_label, n_estimator_max=100, learning_rate_max=1.0)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'bag':
        result, cvtime = Bagging_classifier_with_samples_features_search(
            data_parts_without_label, min_samples=0.1, max_samples=1.0, min_features=0.1, max_features=1.0)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'extratree':
        result, cvtime = Extratrees_with_nEstimators_rate_search(
            data_parts_without_label, n_estimator_max=100)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'gradboost':
        result, cvtime = GradientBoosting_with_nEstimators_rate_search(
            data_parts_without_label, n_estimator_max=100, learning_rate_max=1.0)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'xgboost':
        result, cvtime = XGBoost_with_nEstimators_rate_search(
            data_parts_without_label, n_estimator_max=100)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'linsvc':
        result, cvtime = linSVC_with_tol_iter_search(
            data_parts_without_label, tol_start=-3, tol_end=-7, iter_start=0, iter_end=7)
        search_handler(result, cvtime, parsed_opts)
    elif parsed_opts.A == 'binlr':
        result, cvtime = binLR_with_tol_iter_search(
            data_parts_without_label, tol_start=-3, tol_end=-7, iter_start=0, iter_end=7)
        search_handler(result, cvtime, parsed_opts)
else:
    ratio = parsed_opts.trainpercent
    print(c.OKBLUE, 'Data splitting => ', c.ENDC, 'Total time elapsed',
          str(timedelta(seconds=time()-totaltime)))
    data_parts_without_label = train_test_splitter(data_without_label, ratio)
    if parsed_opts.A not in ('ncentroid', 'rbfsvc', 'qda', 'lda'):
        optimal_params = None
        with open(parsed_opts.resultdir+'cv/'+parsed_opts.A+'/' +
                  str(parsed_opts.D)+'_'+parsed_opts.A+'_'+parsed_opts.S+'.json') as fd:
            cv_content = json.load(fd)
            optimal_params = cv_content['optimal_params']
            if 'max_features' in optimal_params:
                if optimal_params['max_features'] == "sqrt":
                    optimal_params['max_features'] = round(
                        sqrt(data_without_label['X'].shape[1]))
                elif optimal_params['max_features'] == "log2":
                    optimal_params['max_features'] = round(
                        log(data_without_label['X'].shape[1], 2))
                elif optimal_params['max_features'] == None or optimal_params['max_features'] > data_without_label['X'].shape[1]:
                    optimal_params['max_features'] = data_without_label['X'].shape[1]
        print(c.OKBLUE, 'Chosen parameters => ', c.ENDC, optimal_params)
        if parsed_opts.A == 'knn':
            fixed_handler(kNN_with_k_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'dtree':
            fixed_handler(DTree_with_maxFeatures_maxDepth_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'rforest':
            fixed_handler(RForest_with_maxFeatures_maxDepth_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'ada':
            fixed_handler(Adaboost_with_nEstimators_rate_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'bag':
            fixed_handler(Bagging_classifier_with_samples_features_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'extratree':
            fixed_handler(Extratrees_with_nEstimators_rate_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'gradboost':
            fixed_handler(GradientBoosting_with_nEstimators_rate_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'xgboost':
            fixed_handler(XGBoost_with_nEstimators_rate_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'linsvc':
            fixed_handler(linSVC_with_tol_iter_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
        elif parsed_opts.A == 'binlr':
            fixed_handler(binLR_with_tol_iter_fixed(
                data_parts_without_label, optimal_params), parsed_opts)
    if parsed_opts.O is False and parsed_opts.A == 'ncentroid':
        fixed_handler(nCentroid_with_metric_threshold_fixed(data_parts_without_label, {
            'metric': 'manhattan', 'shrink_threshold': None}), parsed_opts)
    if parsed_opts.O is False and parsed_opts.A == 'rbfsvc':
        fixed_handler(rbfSVC_with_C_gamma_fixed(
            data_parts_without_label, {'C': 1.0, 'gamma': 'auto'}), parsed_opts)
    if parsed_opts.O is False and parsed_opts.A == 'lda':
        fixed_handler(LDA_with_tol_iter_fixed(
            data_parts_without_label, {'tol': 1.0e-4}), parsed_opts)
    if parsed_opts.O is False and parsed_opts.A == 'qda':
        fixed_handler(QDA_with_tol_iter_fixed(
            data_parts_without_label, {'tol': 1.0e-4}), parsed_opts)
    print(c.OKBLUE, 'Total time elapsed => ', c.ENDC,
          str(timedelta(seconds=time()-totaltime)))
