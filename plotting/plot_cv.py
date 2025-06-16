#!/usr/bin/env -S python -u
import os
import re
import argparse
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pprint
from collections import Counter

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--directory", action="store",
                    dest="D", help="Which directory to process", type=str)
parsed_opts = parser.parse_args()
data = []
cols = ['day', 'algorithm', 'scaling']
add_cols = []
for root, dirs, files in os.walk(parsed_opts.D):
    for file in files:
        if file.endswith(".json"):
            parts = file.split('.', 1)[0].split('_')            
            with open(os.path.join(root, file), 'r') as fd:
                content = json.load(fd)
                keycount = len(content["optimal_params"].keys())
                for k,v in content["optimal_params"].items():
                    if len(add_cols) < keycount:
                        add_cols.append(k)
                    parts.append(v)
            data.append(parts)

[cols.append(c) for c in add_cols]
df = pd.DataFrame(data=data, columns=cols)
df.replace(to_replace=' None', value=80, inplace=True)
df['day'] = df['day'].astype('int32')
df['algorithm'] = df['algorithm'].astype('str')
df['scaling'] = df['scaling'].astype('str')
try:
    df.iloc[:, 3:] = df.iloc[:, 3:].astype('float64')
except ValueError:
    print("Found a non-numeric value for one of the optimal parameters", df.iloc[:, 3:])

with pd.option_context('display.max_rows', 1, 'display.max_columns', None):
    print(df)
print(df.dtypes)
print(df.describe())

fig, axes = plt.subplots(nrows=3, ncols=2)
axes[-1, -1].axis('off')

fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9,
                    top=0.9, wspace=0.2, hspace=0.6)
fig.suptitle(df['algorithm'][0], fontsize='xx-large')


def plot_bar_from_counter(counter, ax, name):
    pp.pprint(counter)
    frequencies = list(counter.values())
    names = list(counter.keys())
    if isinstance(names[0], float):
        names = [round(elem, 2) if elem > 0.1 else elem for elem in names]
    names = sorted(names)
    print(names)
    x_coordinates = np.arange(len(counter))
    ax.bar(x_coordinates, frequencies, align='center')
    ax.set_title(name)
    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(names))
    return ax


for i, ax in enumerate(axes.flatten()[:-1]):
    print(df.columns[i])
    plot_bar_from_counter(Counter(df[df.columns[i]]), ax, df.columns[i])

plt.show()
