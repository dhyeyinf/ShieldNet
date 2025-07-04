#! /usr/bin/env bash
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 0 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 0 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 0 -A rbfsvc -S No -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 1 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 1 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 1 -A rbfsvc -S No -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 2 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 2 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 2 -A rbfsvc -S No -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 3 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 3 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 3 -A rbfsvc -S No -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 4 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 4 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 4 -A rbfsvc -S No -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 5 -A rbfsvc -S Z -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 5 -A rbfsvc -S MinMax -O True
python3 -u ../ml.py --datadir ../data/CSV/ --resultdir ../results/ -D 5 -A rbfsvc -S No -O True
