import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

train_errors = {f: {i: [] for i in range(5)} for f in function_names}
val_errors = {f: [] for f in function_names}
test_errors = {f: [] for f in function_names}
floating_ops_ee_all = {f: [] for f in function_names}

# loop through the test functions
for test_index, test_function_name in enumerate(function_names):      

    print('test function', test_function_name)

    train_error = {i: [] for i in range(5)}
    val_error = []
    test_error = []

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
        for key in train_error:
            train_error[key].append(df_ee['Train Error '+str(key)].values)

        val_error = df_ee['Validation Error'].values
        test_error = df_ee['Test Error'].values

        floating_ops_ee = df_ee['Number of Floating Point Operations'].values


        for key in train_error:
            train_errors[test_function_name][key] = train_error[key]

        val_errors[test_function_name].append(val_error)
        test_errors[test_function_name].append(test_error)
        floating_ops_ee_all[test_function_name].append(floating_ops_ee)

plt.close('all')
a = 1.2
fig, axes = plt.subplots(nrows=3, ncols=2,
                         figsize=(3.75*a, 5*a),
                         sharey=True, sharex=True)

axes = np.array(axes).flatten()

ordered_function_names = ['Koza-1', 'Nguyen-1', 'Koza-2', 'Nguyen-3', 'Koza-3', 'Nguyen-4']

color_dict = {'Koza-1': 'C2',
              'Koza-2': 'C5',
              'Koza-3': 'C6',
              'Nguyen-1': 'C7',
              'Nguyen-3': 'C8',
              'Nguyen-4': 'C9',
              'Validation': 'C4'}

for i, test_function_name in enumerate(ordered_function_names):

    plt.sca(axes[i])

    # plot equation engineer data
    x, y = gpp.plot_average(floating_ops_ee_all[test_function_name], test_errors[test_function_name],
                            color=color_dict[test_function_name],
                            # label='Test',
                            linewidth=3)

    x, y = gpp.plot_average(floating_ops_ee_all[test_function_name], val_errors[test_function_name],
                            color='C4',
                            # label='Validation',
                            linewidth=2)


    train_order = copy.copy(ordered_function_names)
    train_order.remove(test_function_name)

    for key in train_errors[test_function_name]:
        x, y = gpp.plot_average(floating_ops_ee_all[test_function_name], train_errors[test_function_name][key],
                                color=color_dict[train_order[key]],
                                # label=train_order[key],
                                linewidth=1)

    # Additional plot details
    # plt.title(test_function_name)

    plt.gca().text(.5, .9, test_function_name,
                   horizontalalignment='center',
                   transform=plt.gca().transAxes)

    plt.xticks([i*10**10 for i in range(6)])

plt.subplots_adjust(wspace=0.05, hspace=0.05)

labels = color_dict.keys()
colors = color_dict.values()
custom_lines = [plt.Line2D([0], [0], color=c) for c in colors]
fig.legend(custom_lines, labels,
           ncol=4, handlelength=1,
           handletextpad=0.4, labelspacing=0.25,
           columnspacing=1.3,
           loc='center', bbox_to_anchor=(0.555, 0.96),
           fontsize=8)

plt.subplots_adjust(bottom=0.12, left=0.13, right=0.98, top=0.92)
fig.text(0.02, 0.5, 'Error', ha='center', va='center', rotation=90)
fig.text(0.535, 0.02, 'Computational effort (Num. operations)', ha='center', va='center')
# plt.tight_layout()
plt.savefig(os.path.join(save_path, 'plot_error_on_all_targets.pdf'))

