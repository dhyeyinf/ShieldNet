#!/usr/bin/env -S python -u
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as dts
import datetime
import numpy as np
import pandas as pd
import pprint
import json
import sys
import functools
import operator
from collections import defaultdict
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directory", action="store",
                    dest="D", help="Which directory to process", type=str)
parsed_opts = parser.parse_args()
timings = defaultdict(list)
confmatrices = defaultdict(list)

days = ["HTTP DoS (78M)", "DDoS (603M)", "Bruteforce (390M)", "Brute SSH (333M)", "Bruteforce (39M)", "Infiltration (74M)"]

algorithms = [name for name in os.listdir(
    parsed_opts.D) if os.path.isdir(os.path.join(parsed_opts.D, name))]
for a in algorithms:
    if len(os.listdir(parsed_opts.D+'/'+a)) not in (1, 19):
        print('Data incomplete, aborting!')
        sys.exit(-1)


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


for root, dirs, files in os.walk(parsed_opts.D):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), 'r') as fd:
                content = json.load(fd)
                file_parts = file.split('.')[0].split('_')
                timings[file.split(
                    '.')[0]] = content['estimator']['runtime'][:-3]

tm = list()
for k, v in timings.items():
    file_parts = k.split('_')    
    _ = []
    _.extend(file_parts)
    _.append(v)
    tm.append(_)

timeframe = pd.DataFrame(
    tm, columns=['day', 'algorithm', 'scaling', 'runtime'])
timeframe['day'] = timeframe['day'].astype('int32')
timeframe['algorithm'] = timeframe['algorithm'].astype('str')
timeframe['scaling'] = timeframe['scaling'].astype('str')
timeframe['runtime'] = pd.to_datetime(
    timeframe['runtime'], errors='raise', format="%H:%M:%S.%f", exact=True, utc=True)
timeframe = timeframe.sort_values(by=['day', 'scaling', 'algorithm'])

print("Is null?", timeframe[timeframe['runtime'].isnull()])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(timeframe)
print(timeframe.dtypes)

grouped_frame = timeframe.groupby(['day', 'scaling', 'algorithm'])

latex = timeframe.to_latex(columns=['algorithm', 'day', 'scaling', 'runtime'])
print(latex)

time_fig, time_axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
for i, ax in enumerate(time_axes.flatten()):
    ax.yaxis.set_major_formatter(dts.DateFormatter('%H:%M:%S'))
    ax.set_title(days[i], fontsize='x-large', pad=5.0)
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels(['MinMax', 'No', 'Z'])
    ax.set_xlabel('Scaling method')
    ax.set_ylabel('Run time')
    ax.autoscale(enable=True, axis='both', tight=False)

flattened_axes = time_axes.flatten()

colors = {
    'ada':'#003366',
    'bag':'#3366ff',
    'binlr':'#f4c242',
    'dtree':'#cc0000',
    'knn':'#663300',
    'linsvc':'#003300',
    'ncentroid':'#00ffff',
    'rforest':'#ff9999',
    'rbfsvc':'#42f45c'
}

scales = {
    'MinMax': 1,
    'No': 2, 
    'Z': 3
}

algorithms = ['bag', 'dtree', 'knn', 'ncentroid', 'rforest', 'linsvc', 'binlr', 'rbfsvc']

for day in range(0, 6):
    for scaling in ('MinMax', 'No', 'Z'):
        for index, algorithm in enumerate(algorithms):
            partial_df = grouped_frame.get_group((day, scaling, algorithm)) 
            flattened_axes[day].plot_date(x=scales[scaling]-0.40+0.1*index, y=partial_df.iloc[0, 3], fmt='o', xdate=False, ydate=True, label=algorithm, color=colors[algorithm])
            flattened_axes[day].axvline(x=1.5, linestyle='--', linewidth=0.25, color='grey')
            flattened_axes[day].axvline(x=2.5, linestyle='--', linewidth=0.25, color='grey')


handles, labels = time_axes[1,2].get_legend_handles_labels()
time_axes[1,2].legend(handles[0:len(algorithms)], labels[0:len(algorithms)], bbox_to_anchor=(1.35, 0.50),
                  loc=7, borderaxespad=0., fontsize='x-large')

plt.tight_layout()
plt.show()
