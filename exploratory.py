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
monday_httpdos = parsed_opts.datadir+"TestbedMonJun14Flows.csv"
tuesday_ddos = parsed_opts.datadir+"TestbedTueJun15Flows.csv"
wednesday_bruteforce = parsed_opts.datadir+"TestbedWedJun16Flows.csv"
thursday_bruteforce_ssh = parsed_opts.datadir+"TestbedThuJun17Flows.csv"
saturday_bruteforce = parsed_opts.datadir+"TestbedSatJun12Flows.csv"
sunday_infiltration = parsed_opts.datadir+"TestbedSunJun13Flows.csv"

days = [monday_httpdos, tuesday_ddos, wednesday_bruteforce,
        thursday_bruteforce_ssh, saturday_bruteforce, sunday_infiltration]

chosen_day = days[parsed_opts.D]

                     
def process_time(time):
    time = time.split(" ")[-1]
    h, m = time.split(":")
    return 3600*int(h)+60*int(m)

for chosen_day in days:
    # All columns
    col_names = np.array(["generated", "appName", "totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64",
                      "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF", "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription",
                      "source", "protocolName", "sourcePort", "destination", "destinationPort", "startDateTime", "stopDateTime", "Label"])

    # Read the raw data
    dataframe = pd.read_csv(chosen_day, names=col_names, skiprows=1)
    print(c.OKBLUE, "Reading " + chosen_day + " => ", "Total time elapsed",
          c.ENDC, str(timedelta(seconds=time()-totaltime)))

    # Drop generated col, because it is uninformative
    dataframe.drop(axis=1, columns='generated', inplace=True)
    col_names = col_names[1::]
    #print(c.OKBLUE, "Drop flow-id column => ", "Total time elapsed",
          #c.ENDC, str(timedelta(seconds=time()-totaltime)))

    # Drop payload columns, because I don't know how to make proper features out of them
    dataframe.drop(axis=1, columns=["sourcePayloadAsBase64", "sourcePayloadAsUTF",
                                    "destinationPayloadAsBase64", "destinationPayloadAsUTF"], inplace=True)
    col_names = np.setdiff1d(col_names, np.array(
        ["sourcePayloadAsBase64", "sourcePayloadAsUTF", "destinationPayloadAsBase64", "destinationPayloadAsUTF"]))
    #print(c.OKBLUE, "Drop payload columns => ", "Total time elapsed",
          #c.ENDC, str(timedelta(seconds=time()-totaltime)))

    # Source ip -> numeric
    dataframe["source"] = dataframe["source"].apply(
        lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
    )

    # Destination ip -> numeric
    dataframe["destination"] = dataframe["destination"].apply(
        lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
    )
    #print(c.OKBLUE, "Translate IPs => ", "Total time elapsed",
          #c.ENDC, str(timedelta(seconds=time()-totaltime)))

    dataframe["startDateTime"] = dataframe["startDateTime"].apply(
        lambda time: process_time(time)
    )

    dataframe["stopDateTime"] = dataframe["stopDateTime"].apply(
        lambda time: process_time(time)
    )
    #print(c.OKBLUE, "Process start / stop datetime => ", "Total time elapsed",
    #      c.ENDC, str(timedelta(seconds=time()-totaltime)))


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
        #print(c.OKBLUE, 'Removed tree features => ', 'Total time elapsed',
        #      c.ENDC, str(timedelta(seconds=time()-totaltime)))

    # Translate string attack types to numbers
    attack_dict = {"Normal": 0, "Attack": 1}
    dataframe["Label"] = dataframe["Label"].apply(
        lambda x: attack_dict[x]
    )
    #print(c.OKBLUE, "Binarize label => ", "Total time elapsed",
    #      c.ENDC, str(timedelta(seconds=time()-totaltime)))

    # Print the distribution of the labels
    print(c.OKBLUE, "Label distribution:", c.ENDC)
    print(dataframe['Label'].value_counts())