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
from collections import defaultdict
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directory", action="store",
                    dest="D", help="Which directory to process", type=str)
parser.add_argument("-F", "--filename", action="store", dest="F", help="Path+filename to store file", type=str, required=False)
parsed_opts = parser.parse_args()
data = list()
timings = defaultdict(list)
confmatrices = defaultdict(list)
days = ["HTTP DoS (78M)", "DDoS (603M)", "Bruteforce (390M)", "Brute SSH (333M)", "Bruteforce (39M)", "Infiltration (74M)"]

if len([name for name in os.listdir(parsed_opts.D) if os.path.isfile(os.path.join(parsed_opts.D, name))]) not in (1, 19):
    print("Data incomplete, Aborting!")
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
                filtered_content = removekey(content, 'estimator')
                data.append([file_parts[0], file_parts[1], file_parts[2], filtered_content['accuracy_score'], filtered_content['balanced_accuracy'],
                             filtered_content['f1_score'], filtered_content['precision_score'], filtered_content['recall_score'], filtered_content['roc_auc_score']])
                confmatrices[file_parts[0]].append(
                    {file_parts[2]: filtered_content['confusion_matrix']})

data = np.asarray(data)
df = pd.DataFrame(data, columns=['day', 'algorithm', 'scaling', 'accuracy_score',
                                 'balanced_accuracy', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score'])
df['day'] = df['day'].astype('int32')
df['algorithm'] = df['algorithm'].astype('str')
df['scaling'] = df['scaling'].astype('str')
df.iloc[:, 3:] = df.iloc[:, 3:].astype('float64')
df = df.sort_values(by=['day', 'scaling'])

df.iloc[:, 3:] = df.iloc[:, 3:].round(decimals=4)
latex = df.to_latex(columns=['algorithm','day','scaling','accuracy_score', 'balanced_accuracy', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score' ])
print(latex)

with pd.option_context('display.max_rows', None, 'display.max_columns', 5):
    print(df)
print(df.dtypes)

metric_fig, axes = plt.subplots(2, 3, figsize=(15.0, 12.0))
# axes[-1, -1].axis('off')
# axes[-1, -2].axis('off')

pp.pprint(df.groupby('day').get_group(0))


def autolabel(rects):
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width()/2. + 0.05, 1.01, s='%.3f' %
                float(rect.get_height()), ha='center', va='bottom', rotation=70)


for i, ax in enumerate(axes.flatten()):
    try:
        autolabel(ax.bar(x=np.arange(1, 4)-0.45, height=df.groupby('day').get_group(i)
                         ['accuracy_score'], bottom=0.0, align='edge', width=0.15, color='#ffcc99', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='accuracy_score'))
        autolabel(ax.bar(x=np.arange(1, 4)-0.30, height=df.groupby('day').get_group(i)
                         ['balanced_accuracy'], bottom=0.0, align='edge', width=0.15, color='#ffa64d', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='balanced_accuracy'))
        autolabel(ax.bar(x=np.arange(1, 4)-0.15, height=df.groupby('day').get_group(i)
                         ['f1_score'], bottom=0.0, align='edge', width=0.15, color='#ff8000', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='f1_score'))
        autolabel(ax.bar(x=np.arange(1, 4)+0.00, height=df.groupby('day').get_group(i)
                         ['precision_score'], bottom=0.0, align='edge', width=0.15, color='#b35900', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='precision_score'))
        autolabel(ax.bar(x=np.arange(1, 4)+0.15, height=df.groupby('day').get_group(i)
                         ['recall_score'], bottom=0.0, align='edge', width=0.15, color='#663300', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='recall_score'))
        autolabel(ax.bar(x=np.arange(1, 4)+0.30, height=df.groupby('day').get_group(i)
                         ['roc_auc_score'], bottom=0.0, align='edge', width=0.15, color='#1a0d00', alpha=0.75, edgecolor='#ffffff', linewidth=2, label='roc_auc_score'))
        ax.autoscale(enable=True, tight=False)
        ax.set_xticks(np.arange(1, 4))
        ax.set_xticklabels(['MinMax', 'No', 'Z'])
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('score')
        ax.set_title(days[i], fontsize='large', pad=40.0)
    except KeyError:
        next

handles, labels = axes[1,2].get_legend_handles_labels()
axes[1, 2].legend(handles, labels, bbox_to_anchor=(1.75, 1.25),
                  loc=7, borderaxespad=0., fontsize='large', title=df['algorithm'][0], title_fontsize='x-large')

if parsed_opts.F:
    plt.savefig(fname=parsed_opts.F, dpi='figure', quality=95, orientation='portrait', format='svg')
plt.tight_layout()
plt.show()
