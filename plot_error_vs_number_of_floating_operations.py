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

single_target_plot = False

alg_name = 'TLC-SR'

# experiment number
if single_target_plot:
    exp = 23
else:
    exp = 25

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
genetic_programming = True

if single_target_plot:
    all_at_once = False
    single_target = True
else:
    all_at_once = True
    single_target = False

average_final_test_error = {}

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

# We will plot all the data after
# we loop through and organize it.
# Save all the data in the following
# lists.
error_ee_all = {}
train_error_ee_all = {}
val_error_ee_all = {}
floating_ops_ee_all = {}
error_gp_all = {}
train_error_gp_all = {}
val_error_gp_all = {}
floating_ops_gp_all = {}

pvalues = {}

# loop through the test functions
for test_index, test_function_name in enumerate(function_names):      

    print('test function', test_function_name)

    # The data from the test function. This is the
    # data to be compared using the Mann-Whitney U test
    testdata_ee = {'error': {}, 'floating_ops': {}}
    testdata_gp = {'error': {}, 'floating_ops': {}}
    testdata_sgp = {'error': {}, 'floating_ops': {}}

    error_ee_all[test_function_name] = []
    train_error_ee_all[test_function_name] = []
    val_error_ee_all[test_function_name] = []
    floating_ops_ee_all[test_function_name] = []

    # Since we need to read multiple files
    # to get this data, we will create empty
    # lists for each rep and extend them.
    error_gp_all[test_function_name] = []
    train_error_gp_all[test_function_name] = []
    val_error_gp_all[test_function_name] = []
    floating_ops_gp_all[test_function_name] = []

    # for each repetition.
    for rep in range(nreps):

        # print('rep', rep)

        # First, let's get GP data
        if genetic_programming:

            # Since the GP runs are stored separately, the restart the floating op
            # count. So we need to keep track up it in this script.
            total_floating_ops_gp = 0

            # Adjust the base path
            path_gp = os.path.join(base_data_path, 'gp')

            # Read the data
            if all_at_once:
                # thing = None if test_index == 0 else test_index
                df_gp = pd.read_csv(os.path.join(path_gp, 'best_fitness_data_rep'+str(rep)+'_train_all_at_once_test_index'+str(test_index)+'.csv'))

                # Since the GP runs are stored separately, the restart the floating op
                # count. So we need to keep track up it in this script.
                diff_total_floating_ops_gp = df_gp['Computation'].values[-1]

                error_gp = df_gp['Testing Error'].values
                train_error_gp = df_gp['Training Error'].values
                val_error_gp = df_gp['Validation Error'].values

                floating_ops_gp = total_floating_ops_gp + df_gp['Computation'].values

                # get data for plot
                error_gp_all[test_function_name].append(error_gp)
                train_error_gp_all[test_function_name].append(train_error_gp)
                val_error_gp_all[test_function_name].append(val_error_gp)
                floating_ops_gp_all[test_function_name].append(floating_ops_gp)

                # get test data
                testdata_gp['error'][rep] = error_gp
                testdata_gp['floating_ops'][rep] = df_gp['Computation'].values

                # Increase the total_floating_ops, so the when the floating_ops
                # in the next run of GP starts over, we don't lose track of the
                # total count.
                total_floating_ops_gp += diff_total_floating_ops_gp

            elif single_target:

                df_gp = pd.read_csv(os.path.join(path_gp, 'best_fitness_data_rep'+str(rep)+'_train'+str(test_index)+'_test_index'+str(test_index)+'.csv'))

                error_gp = df_gp['Testing Error'].values
                train_error_gp = df_gp['Training Error'].values
                val_error_gp = df_gp['Validation Error'].values

                floating_ops_gp = df_gp['Computation'].values

                # get data for plot
                error_gp_all[test_function_name].append(error_gp)
                train_error_gp_all[test_function_name].append(train_error_gp)
                val_error_gp_all[test_function_name].append(val_error_gp)
                floating_ops_gp_all[test_function_name].append(floating_ops_gp)

                # get test data
                testdata_gp['error'][rep] = error_gp
                testdata_gp['floating_ops'][rep] = df_gp['Computation'].values


            else:
                
                # Let's loop over all the training functions.
                # By training function, I mean the function that correspond
                # to the training set.
                for train_target_num in range(num_train_targets):

                    df_gp = pd.read_csv(os.path.join(path_gp, 'best_fitness_data_rep'+str(rep)+'_train'+str(train_target_num)+'.csv'))

                    # Since the GP runs are stored separately, the restart the floating op
                    # count. So we need to keep track up it in this script.
                    diff_total_floating_ops_gp = df_gp['Computation'].values[-1]

                    error_gp = df_gp['Testing Error'].values
                    train_error_gp = df_gp['Training Error'].values
                    val_error_gp = df_gp['Validation Error'].values

                    floating_ops_gp = total_floating_ops_gp + df_gp['Computation'].values

                    # get data for plot
                    error_gp_all[test_function_name].append(error_gp)
                    train_error_gp_all[test_function_name].append(train_error_gp)
                    val_error_gp_all[test_function_name].append(val_error_gp)
                    floating_ops_gp_all[test_function_name].append(floating_ops_gp)

                    # get test data
                    testdata_gp['error'][rep] = error_gp
                    testdata_gp['floating_ops'][rep] = df_gp['Computation'].values

                    # Increase the total_floating_ops, so the when the floating_ops
                    # in the next run of GP starts over, we don't lose track of the
                    # total count.
                    total_floating_ops_gp += diff_total_floating_ops_gp

        # Next, let's get the data for the equation engineer
        if equation_engineer:

            path_ee = base_data_path

            # Equation engineer trains with multiple target functions at once, so there is no
            # switching here. Thus, no for loop. But, the training and test information is split
            # between two different files. Let's start with the training.
            # df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_rep'+str(rep)+'_'+test_function_name+'.csv'))
            df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_rep'+str(rep)+'_test_index'+str(test_index)+'.csv'))


            # Get data from training period.
            error_ee = df_ee['Test Error'].values
            if single_target_plot:
                train_error_ee = df_ee['Train Error Sum'].values
            else:
                train_error_ee = df_ee['Train Error Average'].values
            val_error_ee = df_ee['Validation Error'].values
            floating_ops_ee = df_ee['Number of Floating Point Operations'].values

            # Get plot data. Last point is the extension of EE to the total amound of allowed
            # floating ops even though EE was not run that long.
            # error_ee_all.append(np.hstack((error_ee, error_ee_test, [error_ee_test[-1]])))
            # floating_ops_ee_all.append(np.hstack((floating_ops_ee, floating_ops_ee_test, [2*10**10])))
            error_ee_all[test_function_name].append(error_ee)
            train_error_ee_all[test_function_name].append(train_error_ee)
            val_error_ee_all[test_function_name].append(val_error_ee)
            floating_ops_ee_all[test_function_name].append(floating_ops_ee)

            # Get test data for stat test
            testdata_ee['floating_ops'][rep] = floating_ops_ee[-1] - floating_ops_ee[0]
            testdata_ee['error'][rep] = error_ee

    if testdata_ee['error']:
        # Time for stats: compare (p-value) test values at max floating_ops for EE
        ee_test_final = [testdata_ee['error'][rep][-1] for rep in testdata_ee['error']]

        # Take the max number of floating ops during testing of EE to use for the comparison.
        floating_ops_level = np.max([testdata_ee['floating_ops'][rep] for rep in testdata_ee['floating_ops']])

        print('floating_ops_level', floating_ops_level)
    
    # This if statement, is here so commenting out
    # the gp data collection does not break the script.
    if testdata_gp['error']:

        # Use the indices to get the error
        gp_test_final = [testdata_gp['error'][rep][-1] for rep in testdata_gp['error']]

        # Do the test: Is EE < GP in terms of test error for given amount of floating ops?
        stat, pvalue = scipy.stats.mannwhitneyu(ee_test_final, gp_test_final, alternative='less')
        print(test_function_name, 'ee < gp', pvalue)
        pvalues[test_function_name] = pvalue

        average_final_test_error[test_function_name] = np.mean(ee_test_final)

        # Update p-value table
        table = table.append({'Target': test_function_name, 'Test': 'ee < gp', 'p-value': pvalue, 'Significance Level': sig_level}, ignore_index=True)

        # If the p-value is significant,
        # bold the row.
        if pvalue < sig_level:

            current_row = table.shape[0] - 1
            bold_rows.append(current_row)

    # New plot of the data actually being compared.
    plt.close('all')
    plt.figure()

    if testdata_ee['error']:
        gpp.plot_emprical_cumulative_distribution_funtion(ee_test_final, label='ee')

    # This if statement, is here so commenting out
    # the gp data collection does not break the script.
    if testdata_gp['error']:
        gpp.plot_emprical_cumulative_distribution_funtion(gp_test_final, label='gp')

    # plt.xscale('log')
    plt.xlabel('Final Test Error')
    plt.title(test_function_name)

    # Remove duplicat labels in the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(os.path.join(save_path, 'ee_gp_floating_ops_cdf_'+test_function_name+'.pdf'))

# save the table
gpdm.write_table_with_bold_rows(df=table, filename=os.path.join(stats_save_path, 'stats_ee_gp_floating_ops_exp'+str(exp)), bold_rows=bold_rows)

print('average final test error', average_final_test_error)

plt.close('all')
width = 7.5
print((width, width*4.8/6.4))
# height = 5
if single_target_plot:
    height = 4.5
else:
    height = 5
a = 1.0
fig, axes = plt.subplots(nrows=2, ncols=3,
                         figsize=(width*a, height*a),
                         sharey='row', sharex=True)

axes = np.array(axes).flatten()

ordered_function_names = ['Koza-1', 'Koza-2', 'Koza-3', 'Nguyen-1', 'Nguyen-3', 'Nguyen-4']

for i, test_function_name in enumerate(ordered_function_names):

    plt.sca(axes[i])

    # Time for the plot
    # We want to average the repetition together, however, the number of 
    # floating ops is not exactly the same between different runs, so we 
    # must get rep matrices to have consistent run length (row length, number of columns).
    # This is done with function above. So is the actual plotting. These operations can take
    # some time if there is a lot of data, so we have shorted the number of data points in GP
    # to speed this up.
    # floating_ops_gp_all = [x for x in floating_ops_gp_all[test_function_name]]
    # error_gp_all = [x for x in error_gp_all[test_function_name]]
    # train_error_gp_all = [x for x in train_error_gp_all[test_function_name]]

    # plt.close('all')
    # plt.figure('train')
    # plt.figure('validation')
    # plt.figure('test', figsize=(7.5/2, 7.5/2*4.8/6.4))

    if equation_engineer:
        
        print('plot ee')

        # plot equation engineer data
        x, y = gpp.plot_standard_error(floating_ops_ee_all[test_function_name], error_ee_all[test_function_name], color='C0', label=alg_name)

        # plt.figure('train')
        # x, y = plot_confidence_interval(floating_ops_ee_all, train_error_ee_all, color='C0', label='EE')

        # plt.figure('validation')
        # x, y = plot_confidence_interval(floating_ops_ee_all, val_error_ee_all, color='C0', label='EE')


    if genetic_programming:

        print('plot gp')

        floating_ops_gp = [x[::99] for x in floating_ops_gp_all[test_function_name]]
        error_gp = [x[::99] for x in error_gp_all[test_function_name]]

        # This time we want the averages so that we can plot the training switches. I don't want
        # to draw vertical lines because equation engineer does not switch training. Maybe I should
        # have it switch, so it is more similar? Start with one target and add one at every switch?
        x, y = gpp.plot_standard_error(floating_ops_gp, error_gp, color='C1', label='GP')

        # plt.figure('train')
        # train_error_gp_all = [x[::100] for x in train_error_gp_all]

        # x, y = plot_confidence_interval(floating_ops_gp_all, train_error_gp_all, color='C1', label='GP')

        # plt.figure('validation')
        # val_error_gp_all = [x[::100] for x in val_error_gp_all]

        # x, y = plot_confidence_interval(floating_ops_gp_all, val_error_gp_all, color='C1', label='GP')


    if single_target_plot:
        if i == 5:
            plt.ylim([-0.03, 1.0])
    # plt.ylim([-.03, 1.5])
    plt.xticks(np.linspace(0, 15*10**10, 4))
    # plt.xticks(np.linspace(0, 5*10**10, 6))


    switches = np.arange(0, max_train_switches+1, train_switch_step)
    x = []
    y = []

    # for fig_name in ['train', 'validation', 'test']:

    ylim = plt.ylim()

    if plot_train_switch:
        for s in switches:

            plt.plot([s]*2, ylim, '--k', zorder=0, label='Training Switch')

    if single_target_plot:
        if i == 3:
            loc = 'center right'

            # Remove duplicat labels in the legend.
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc=loc,
                       handlelength=1)

    else:
        if i == 0:
            loc = 'lower left'

            # Remove duplicat labels in the legend.
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc=loc,
                       handlelength=1)

    # Additional plot details
    # plt.title(test_function_name)

    plt.gca().text(.5, .9, test_function_name,
                   horizontalalignment='center',
                   transform=plt.gca().transAxes)

    if not single_target_plot and genetic_programming and equation_engineer:
        pval_yscale = 0.8 if i < 5 else 0.1

        if pvalues[test_function_name] < 0.001:
            pval_cond = '< 0.001**'
        elif pvalues[test_function_name] < 0.01:
            pval_cond = '< 0.01*'
        elif pvalues[test_function_name] < 0.05:
            pval_cond = '< 0.05*'
        else:
            pval_cond = '= %.3f' % pvalues[test_function_name]


        plt.gca().text(.5, pval_yscale, '$p$('+alg_name+' < GP) '+pval_cond,
                       horizontalalignment='center',
                       transform=plt.gca().transAxes)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.subplots_adjust(left=0.08, right=0.99, top=0.98)

if single_target_plot:
    plt.subplots_adjust(bottom=0.13)
else:
    plt.subplots_adjust(bottom=0.12)

fig.text(0.02, 0.5, 'Test error', ha='center', va='center', rotation=90)
fig.text(0.54, 0.02, 'Computational effort (Num. operations)', ha='center', va='center')

# plt.tight_layout()
if single_target_plot:
    filename = 'plot_error_vs_number_of_floating_operations_v3_single_training_dataset.pdf'
else:
    filename = 'plot_error_vs_number_of_floating_operations_v3_multi_training_datasets.pdf'
plt.savefig(os.path.join(save_path, filename))
