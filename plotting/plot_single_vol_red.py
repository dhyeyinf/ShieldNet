#!/usr/bin/env -S python -u
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import Button, RadioButtons
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
parser.add_argument("-F", "--filename", action="store", dest="F",
                    help="Path+filename to store file", type=str, required=False)
parsed_opts = parser.parse_args()
data = list()
timings = defaultdict(list)
confmatrices = defaultdict(list)
days = ['ISCX HTTP-DoS 81.5 MB', 'ISCX DDoS 631.8 MB', 'ISCX bruteforce 408.8 MB (Do NOT use only 11 malicious samples out of 522263 samples)',
        'ISCX SSH bruteforce 349.0 MB', 'ISCX bruteforce 40.3 MB', 'ISCX infiltration 77.5 MB']

algorithm = parsed_opts.D.split('/')[-1].split('_')[-1]

# if len([name for name in os.listdir(parsed_opts.D) if os.path.isfile(os.path.join(parsed_opts.D, name))]) not in (1, 4950, 4951, 3300, 3301):
#     print("Data incomplete, Aborting!")
#     sys.exit(-1)


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


for root, dirs, files in os.walk(parsed_opts.D):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), 'r') as fd:
                content = json.load(fd)
                file_parts = file.rsplit('.', 1)[0].split('_')
                timings[file.rsplit('.', 1)[0]
                        ] = content['estimator']['runtime'][:-3]
                filtered_content = removekey(content, 'estimator')
                data.append([file_parts[0], file_parts[1], file_parts[2], filtered_content['reduced_by'], filtered_content['train_percent'], filtered_content['accuracy_score'], filtered_content['balanced_accuracy'],
                             filtered_content['f1_score'], filtered_content['precision_score'], filtered_content['recall_score'], filtered_content['roc_auc_score']])
                confmatrices[file_parts[0]].append(
                    {file_parts[2]: filtered_content['confusion_matrix']})

data = np.asarray(data)
df = pd.DataFrame(data, columns=['day', 'algorithm', 'scaling', 'reduced_by', 'train_percent', 'accuracy_score',
                                 'balanced_accuracy', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score'])
df['day'] = df['day'].astype('int32')
df['algorithm'] = df['algorithm'].astype('str')
df['scaling'] = df['scaling'].astype('str')
df.iloc[:, 3] = df.iloc[:, 3].astype('int64')
df.iloc[:, 4:] = df.iloc[:, 4:].astype('float64')
df = df.sort_values(by=['day', 'scaling', 'reduced_by', 'train_percent'])

df.iloc[:, 4:] = df.iloc[:, 4:].round(decimals=4)
latex = df.to_latex(columns=['algorithm', 'day', 'scaling', 'reduced_by', 'train_percent', 'accuracy_score',
                             'balanced_accuracy', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score'])
# print(latex)

with pd.option_context('display.max_rows', None, 'display.max_columns', 5):
    print(df)
print(df.dtypes)
print(df.columns)

metric_fig, axes = plt.subplots(1, 1, figsize=(15.0, 15.0))

# algo_ax = plt.axes([0.9, 0.4, 0.15, 0.15], facecolor='white')
# algo_radio = RadioButtons(algo_ax, ('dtree', 'bagging', 'ada', 'rforest', 'extratree', 'gradboost', 'xgboost', 'knn', 'ncentroid', 'binlr', 'linsvc', 'rbfsvc'))
#
day_selected = 0
day_ax = plt.axes([0.91, 0.6, 0.075, 0.1], facecolor='white')
day_radio = RadioButtons(day_ax, df['day'].unique())


def day_radio_click(label):
    global day_selected
    day_selected = int(label)


day_radio.on_clicked(day_radio_click)

scaling_selected = 'Z'
scaling_ax = plt.axes([0.91, 0.4, 0.075, 0.1], facecolor='white')
scaling_radio = RadioButtons(scaling_ax, ('Z', 'MinMax', 'No'))


def scaling_radio_click(label):
    global scaling_selected
    scaling_selected = label


scaling_radio.on_clicked(scaling_radio_click)

reduce_selected = 0
reduce_ax = plt.axes([0.91, 0.2, 0.075, 0.1], facecolor='white')
reduce_radio = RadioButtons(reduce_ax, df['reduced_by'].unique())


def reduce_radio_click(label):
    global reduce_selected
    reduce_selected = int(label)


reduce_radio.on_clicked(reduce_radio_click)

update_ax = plt.axes([0.925, 0.01, 0.05, 0.05])
update_button = Button(update_ax, 'Redraw')


def update_plot(event):
    axes.clear()
    produce_plot(algorithm, day_selected, scaling_selected, reduce_selected)
    if event.inaxes is not None:
        event.inaxes.figure.canvas.draw_idle()


update_button.on_clicked(update_plot)

df = df.sort_values(['day', 'train_percent'], ascending=[True, True])

grouped = df.groupby(['algorithm', 'day', 'scaling', 'reduced_by'])
#pp.pprint(grouped.get_group((algorithm, 0, 'Z', 0))['train_percent'].values)
#pp.pprint(grouped.get_group((algorithm, 0, 'Z', 0))['accuracy_score'].values)


def produce_plot(algo, day, scaling, reduc):
    axes.set_ylabel('Score', fontsize=14)
    axes.set_xlabel('Volume of dataset used for training', fontsize=12)
    if algorithm in ('ada', 'bag', 'rforest', 'dtree', 'knn', 'ncentroid', 'xgboost', 'gradboost', 'extratree'):
        gp = grouped.get_group((algo, day, scaling, reduc))
        x = gp['train_percent'].values
        acc = gp['accuracy_score'].values
        bacc = gp['balanced_accuracy'].values
        f1 = gp['f1_score'].values
        precision = gp['precision_score'].values
        recall = gp['recall_score'].values
        roc_auc = gp['roc_auc_score'].values
        print("Day", days[day], "Scaling", scaling,
              "Reduction", reduc, "maxima per metric")
        print("accuracy_score", gp['accuracy_score'].max())
        print("balanced_accuracy", gp['balanced_accuracy'].max())
        print("f1_score", gp["f1_score"].max())
        print("precision_score", gp["precision_score"].max())
        print("recall_score", gp["recall_score"].max())
        print("roc_auc", gp["roc_auc_score"].max(), "\n")
        axes.plot(x, acc, linestyle='-', marker='x', markersize=7.5,
                  color='#a9a9a9', label='accuracy')
        axes.plot(x, bacc, linestyle='-', marker='x', markersize=7.5,
                  color='#000000', label='balanced_accuracy')
        axes.plot(x, f1, linestyle='-', marker='x', markersize=7.5,
                  color='#e6194b', label='F1')
        axes.plot(x, precision, linestyle='-', marker='x', markersize=7.5,
                  color='#f58231', label='precision')
        axes.plot(x, recall, linestyle='-', marker='x', markersize=7.5,
                  color='#42d4f4', label='recall')
        axes.plot(x, roc_auc, linestyle='-', marker='x', markersize=7.5,
                  color='#911eb4', label='roc_auc')
        axes.set_title(days[day], fontsize='large', pad=40.0)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles, labels, bbox_to_anchor=(0.95, 0.15), loc=7,
                    borderaxespad=0., fontsize='large', title=algorithm, title_fontsize='x-large')
    elif algorithm in ('binlr', 'linsvc', 'rbfsvc'):
        gp = grouped.get_group((algo, day, scaling, reduc))
        x = gp['train_percent'].values
        acc = gp['accuracy_score'].values
        bacc = gp['balanced_accuracy'].values
        f1 = gp['f1_score'].values
        precision = gp['precision_score'].values
        recall = gp['recall_score'].values
        roc_auc = gp['roc_auc_score'].values
        print("Day", days[day], "Scaling", scaling,
              "Reduction", reduc, "maxima per metric")
        print("accuracy_score", gp['accuracy_score'].max())
        print("balanced_accuracy", gp['balanced_accuracy'].max())
        print("f1_score", gp["f1_score"].max())
        print("precision_score", gp["precision_score"].max())
        print("recall_score", gp["recall_score"].max())
        print("roc_auc", gp["roc_auc_score"].max(), "\n")
        axes.plot(x, acc, '-', marker='x', markersize=7.5,
                  color='#a9a9a9', label='accuracy')
        axes.plot(x, bacc, '-', marker='x', markersize=7.5,
                  color='#000000', label='balanced_accuracy')
        axes.plot(x, f1, '-', marker='x', markersize=7.5,
                  color='#e6194b', label='F1')
        axes.plot(x, precision, '-', marker='x', markersize=7.5,
                  color='#f58231', label='precision')
        axes.plot(x, recall, '-', marker='x', markersize=7.5,
                  color='#42d4f4', label='recall')
        axes.plot(x, roc_auc, '-', marker='x', markersize=7.5,
                  color='#911eb4', label='roc_auc')
        axes.set_title(days[day], fontsize='large', pad=40.0)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles, labels, bbox_to_anchor=(0.95, 0.15), loc=7,
                    borderaxespad=0., fontsize='large', title=algorithm, title_fontsize='x-large')

    # if parsed_opts.F:
    #     plt.savefig(fname=parsed_opts.F, dpi='figure', quality=95,
    #                 orientation='portrait', format='svg')


produce_plot(algorithm, 0, 'Z', 0)
plt.show()
