#! /usr/bin/env python

import os
import numpy as np
# The default umask is 0o22 which turns off write permission of group and others
os.umask(0)
for algo in ("knn", "ncentroid", "dtree", "rforest", "ada", "bag", "xgboost", "gradboost", "extratree"):
    for day in range(0, 6):
        with open(
                os.open("runner_single_" + algo + "_" + str(day) + ".sh",
                        os.O_CREAT | os.O_WRONLY, 0o744), 'w') as fh:
            commands = "#! /usr/bin/env bash\n"
            for scaling in ("Z", "MinMax", "No"):
                for reduceby in (0, 1, 2, 3, 4):
                    for trainpercent in np.linspace(0.01, 0.50,
                                                    11).round(decimals=2):
                        commands += "python3 -u ../ml.py -D " + str(
                            day
                        ) + " -A " + algo + " -S " + scaling + " --reduceby " + str(
                            reduceby
                        ) + " --trainpercent " + str(
                            trainpercent
                        ) + " --datadir ../data/CSV/ --resultdir ../results/ --disk 1 --export 1\n"
            fh.write(commands)
            fh.flush()

for algo in ("linsvc", "rbfsvc", "binlr"):
    for day in range(0, 6):
        with open(
                os.open("runner_single_" + algo + "_" + str(day) + ".sh",
                        os.O_CREAT | os.O_WRONLY, 0o744), 'w') as fh:
            commands = "#! /usr/bin/env bash\n"
            for scaling in ("Z", "MinMax"):
                for reduceby in (0, 1, 2, 3, 4):
                    for trainpercent in np.linspace(0.01, 0.50,
                                                    11).round(decimals=2):
                        commands += "python3 -u ../ml.py -D " + str(
                            day
                        ) + " -A " + algo + " -S " + scaling + " --reduceby "+str(reduceby)+" --trainpercent " + str(
                            trainpercent
                        ) + " --datadir ../data/CSV/ --resultdir ../results/ --disk 1 --export 1\n"
            fh.write(commands)
            fh.flush()
