#! /usr/bin/env python

import os
# The default umask is 0o22 which turns off write permission of group and others
os.umask(0)
for algo in ("knn", "ncentroid", "dtree", "rforest", "ada", "bag", "xgboost", "gradboost", "extratree"):
    with open(
            os.open("runner_cv_" + algo + ".sh", os.O_CREAT | os.O_WRONLY,
                    0o744), 'w') as fh:
        commands = "#! /usr/bin/env bash\n"
        for day in range(0, 6):
            for scaling in ("Z", "MinMax", "No"):
                commands += "python3 -u ../ml.py -D " + str(
                    day
                ) + " -A " + algo + " -S " + scaling + " -O True --datadir ../data/CSV/ --resultdir ../results/\n"
        fh.write(commands)
        fh.flush()

for algo in ("linsvc", "binlr"):
    with open(
            os.open("runner_cv_" + algo + ".sh", os.O_CREAT | os.O_WRONLY,
                    0o744), 'w') as fh:
        commands = "#! /usr/bin/env bash\n"
        for day in range(0, 6):
            for scaling in ("Z", "MinMax"):
                commands += "python3 -u ../ml.py -D " + str(
                    day
                ) + " -A " + algo + " -S " + scaling + " -O True --datadir ../data/CSV/ --resultdir ../results/\n"
        fh.write(commands)
        fh.flush()
