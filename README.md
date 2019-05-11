# Genetic Programming

This is the readme file for GeneticProgrammingAfpo. This was created by Ryan Grindle. This is supposed to be the basic version of GP since I will be implementing various experiments that alter the functionality used here. This was build using python3.5 and python3.6

Parameters for a run of GP are given as command line arguments. Some information can be found by doing `python3 run_gp.py <rep> <tar> <exp>` where `<rep>` is a non-negative integer specify which repetition is being run. This number will be used to seed the random number generator that generates/picks dataset. `<tar>` is string that refers to a target function (or a dataset). In consts.py there the string can be mapped to the necessary data. `<exp>` is the experiment number. The experiment number is used to seed the random number generator and becomes part of the path to the output files. Data generated during the run will be stored in
`$GP_DATA/<exp>/<tar>/`
There may be another folder specifying more details of the experiment.

Output data includes summary statistics of all fitness objective during evolution. It also includes data describing all individuals in the final generation.

OTHER OPTIONS....

# AFPO
AFPO --- more info here...

## Python Packages Used
* numpy
* pandas
* pygraphviz
* scipy