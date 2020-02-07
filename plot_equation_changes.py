import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pylab as pylab
params = {#'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         # 'axes.labelsize': 'x-large',
         'axes.titlesize':'small',}
pylab.rcParams.update(params)

import general_plotting.plotting as gpp
import general_plotting.data_manipulation as gpdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import os
import copy
from collections import OrderedDict

plot_train_switch = False
train_switch_step = 2*10**9
# num_train_switches = 5
max_train_switches = 10**10

alg_name = 'TLC-SR'

# experiment number
exp = 24

# number of runs (repetitions)
nreps = 30

# How many targets are used during
# training.
num_train_targets = 5

# significance level for mann-whitney U test
# Bonferroni correction is applied since we
# compare equation engineer to GP and to 
# semantic GP.
sig_level = 0.05

# Flags to include gp and semantic gp in the
# plots, or not.
equation_engineer = True

all_at_once = True

base_data_path = os.path.join(os.environ['EE_DATA'], 'experiment'+str(exp))

# save locations for figures and statistics
# create the folders if necessary
save_path = os.path.join(base_data_path, 'figures')
stats_save_path = os.path.join(base_data_path, 'stats')

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(stats_save_path):
    os.makedirs(stats_save_path)

# table will have all p-values in it
table = pd.DataFrame([], columns=['Target', 'Test', 'p-value', 'Significance Level'])

# bold_rows will have the indices of rows that
# have significant p-values. This rows will be
# bold when saved.
bold_rows = []

# function_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
#                   'keijzer15', 'r1', 'r2', 'r3']
# function_names = ['koza1']
function_names = ['Nguyen-4', 'Koza-1', 'Koza-2', 'Koza-3', 'Nguyen-1', 'Nguyen-3']

unique_counts = {}
total_counts = {}
floating_ops_ee_all = {}


# loop through the test functions
for test_index, test_function_name in enumerate(function_names):      

    print('test function', test_function_name)

    # We will plot all the data after
    # we loop through and organize it.
    # Save all the data in the following
    # lists.
    unique_counts[test_function_name] = []
    total_counts[test_function_name] = []
    floating_ops_ee_all[test_function_name] = []

    # for each repetition.
    for rep in range(nreps):

        print('rep', rep)

        path_ee = base_data_path

        # Equation engineer trains with multiple target functions at once, so there is no
        # switching here. Thus, no for loop. But, the training and test information is split
        # between two different files. Let's start with the training.
        # df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_rep'+str(rep)+'_'+test_function_name+'.csv'))
        df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_rep'+str(rep)+'_test_index'+str(test_index)+'.csv'))


        # Get data from training period.
        unique_count = df_ee['Number of Unique Validation Errors'].values
        total_count = df_ee['Number of Validation Errors'].values
        floating_ops_ee = df_ee['Number of Floating Point Operations'].values

        # Get plot data. Last point is the extension of EE to the total amound of allowed
        # floating ops even though EE was not run that long.
        # error_ee_all.append(np.hstack((error_ee, error_ee_test, [error_ee_test[-1]])))
        # floating_ops_ee_all.append(np.hstack((floating_ops_ee, floating_ops_ee_test, [2*10**10])))
        unique_counts[test_function_name].append(unique_count)
        total_counts[test_function_name].append(total_count)
        floating_ops_ee_all[test_function_name].append(floating_ops_ee)


plt.close('all')
a = 1.2
fig, axes = plt.subplots(nrows=3, ncols=2,
                         figsize=(3.75*a, 5*a),
                         sharey=True, sharex=True)

axes = np.array(axes).flatten()

ordered_function_names = ['Koza-1', 'Nguyen-1', 'Koza-2', 'Nguyen-3', 'Koza-3', 'Nguyen-4']

for i, test_function_name in enumerate(ordered_function_names):

    plt.sca(axes[i])

    # plot equation engineer data
    x, y = gpp.plot_average(floating_ops_ee_all[test_function_name], total_counts[test_function_name], color='k', label='Total')
    x, y = gpp.plot_average(floating_ops_ee_all[test_function_name], unique_counts[test_function_name], color='C3', label='Unique')

    # Additional plot details
    plt.title(test_function_name, fontsize=8)

    # if i > 2:
    #     plt.xlabel('FLoPs')

    if i == 0:
        plt.legend(loc='center left', handlelength=1, fontsize=8)

    # plt.ylim(bottom=0)
    plt.xticks([i*10**10 for i in range(6)])

plt.subplots_adjust(bottom=0.12, left=0.15, right=0.98, top=0.9, wspace=0.05, hspace=0.25)
fig.text(0.02, 0.5, 'Average number of equations', ha='center', va='center', rotation=90)

fig.text(0.5, 0.02, 'Computational effort (Num. operations)', ha='center', va='center')
# plt.tight_layout()
plt.savefig(os.path.join(save_path, 'plot_equation_changes.pdf'))

# fig, axes = plt.subplots(nrows=2, ncols=3,
#                          figsize=(3.75, 3.75*3/2),
#                          sharey=True, sharex=True)

# axes = np.array(axes).flatten()

# for i, test_function_name in enumerate(ordered_function_names):

#     plt.sca(axes[i])

#     fraction_counts = [u/t for u, t in zip(unique_counts[test_function_name], total_counts[test_function_name])]

#     x, y = gpp.plot_confidence_interval(floating_ops_ee_all[test_function_name], fraction_counts, color='C2', label='Fraction')

#     plt.title(test_function_name, fontsize='small')


# # plt.ylim([0, 1])
# # plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'plot_equation_changes_fraction.pdf'))

