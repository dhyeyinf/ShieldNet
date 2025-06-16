# Analysis of ISCXIDS2012 through supervised learners
This repository contains the source code and results of an extensive analysis of the ISCXIDS2012 intrusion detection data set.
The results and conclusions are described in detail in an upcoming publication titled: "Classification hardness for supervised learners on 20 years of intrusion detection data".

## Visualization
Interactive plots are available through the plot_single_vol_red.py script in the plotting library. 
The radiobuttons allow to change parameters after which a redraw through the button is necessary to update the plot.
Everything that is required to get this up and running is in the code block below.

```bash
apt-get update
apt-get install python3 python3-pip
ln -s -f /usr/bin/python3 /usr/bin/python
pip3 install --user numpy pandas matplotlib sklearn
./plotting/plot_single_vol_red.py -D results/single/dtree
```
If python3 is the default python version for your distribution, it is not necessary to perform the installation of python and pip or to symlink the binary.