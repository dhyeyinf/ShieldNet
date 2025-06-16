#!/usr/bin/env -S python -u
import os
import argparse
import datetime
import json
import sys
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directory", action="store", dest="D", help="Which directory to process", type=str)
parsed_opts = parser.parse_args()

algorithm = parsed_opts.D.split('/')[-1]

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def generate_required_list(algorithm):
    filenames = set()
    if algorithm in ("knn", "ncentroid", "dtree", "rforest", "ada", "bag", "gradboost", "extratree", "xgboost"):
         for day in range(0, 6):
            for scaling in ("Z", "MinMax", "No"):
                #for trainpercent in np.linspace(0.01, 0.50, 11).round(decimals=2):
                for trainpercent in (0.001, 0.005):
                    for reduceby in (0, 1, 2, 3, 4):
                       filename = str(day)+"_"+algorithm+"_"+scaling+"_"+str(trainpercent)+"_"+str(reduceby)+".json"
                       filenames.add(filename)

    elif algorithm in ("linsvc", "rbfsvc", "binlr"):
        for day in range(0, 6):
            for scaling in ("Z", "MinMax"):
                #for trainpercent in np.linspace(0.01, 0.50, 11).round(decimals=2):
                for trainpercent in (0.001, 0.005):
                    for reduceby in (0, 1, 2, 3, 4):
                       filename = str(day)+"_"+algorithm+"_"+scaling+"_"+str(trainpercent)+"_"+str(reduceby)+".json"
                       filenames.add(filename)
    else:
        print("Unknown algorithm")

    return filenames

found_in_dir = set()
for root, dirs, files in os.walk(parsed_opts.D):
    for file in files:
        if file.endswith(".json"):
            found_in_dir.add(file)

print("Missing files")
print("#! /usr/bin/env bash")
for b in generate_required_list(algorithm).difference(found_in_dir):
    parts = b.split('_')
    print("python3 -u ../ml.py -D "+parts[0]+" -A "+ parts[1]+" -S "+parts[2]+" --trainpercent "+parts[3]+" --reduceby "+parts[4].split('.')[0]+" --datadir ../data/CSV/ --resultdir ../results/")
    
