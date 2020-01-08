import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import os
import copy
from collections import OrderedDict

def is_number(x):
    """Check if x is a number.

    Parameters
    ----------
    x : str
        A string that may or may not
        be a number.

    Returns
    -------
    bool
        True if number. Else False.
    """

    try:
        float(x)   # this will fail if not a number
        return True

    except ValueError:
        return False


def remove_duplicates(x1, x2):
    """Take two equal length lists where x1 has many duplicate values
    find the first instance of a duplicate and the last. Remove the
    intermediate values from both lists.

    Parameters
    ----------
    x1, x2 : lists
        x1 is the one with duplicates to remove

    Returns
    -------
    x1, x2 : lists
        Subsets of the original x1 and x2.
    """

    assert len(x1) == len(x2), 'x1 and x2 must be the same length'

    # This list of indices will be the elements of x1 and x2
    # to be kept.
    indices = []

    # prev is the previous value in the list
    prev = None

    # duplicates is true if the loop is currently
    # in a sequence of duplicate values.
    duplicates = False

    for i, x in enumerate(x1):

        # if first element of x1
        if prev is None:

            # keep this element
            indices.append(i)

        else:

            # if we find a duplicate
            if x == prev:
                duplicates = True
                continue

            else:

                # if sequence of duplicates has ended
                if duplicates:

                    # keep this element
                    indices.append(i-1)
                    duplicates = False

                # keep this element
                indices.append(i)

        # update previous element
        prev = x

    indices = np.array(indices)

    return x1[indices], x2[indices]


def get_emprical_cumulative_distribution_funtion(X):
    """Get the probability that x_i > X after X has
    been sorted (that is x_i is >= i+1 other x's).

    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF

    Returns
    -------
    p : list
        A list of the same length as X that
        give the probability that x_i > X
        where X is a randomly selected value
        and i is the index.
    """

    X_sorted = sorted(X)

    n = len(X)

    p = [i/n for i, x in enumerate(X)]

    return p, X_sorted


def plot_emprical_cumulative_distribution_funtion(X, labels=True, label=None, color=None):
    """Use get_emprical_cumulative_distribution_funtion to plot the CDF.

    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF
    labels : bool (default=True)
        If true, label x-axis x and y-axis Pr(X < x)
    label : str (default=None)
        The legend label.
    color : str (default=None)
        Color to used in plot. If none, it will not
        be pasted to plt.step.
    """

    p, X = get_emprical_cumulative_distribution_funtion(X)

    if color is None:
        plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4, label=label)

    else:
        plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4, label=label, color=color)

    if labels:

        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')


def write_table_with_bold_rows(df, filename, bold_rows):
    """Write a table to a .xls file with rows bolded.

    Parameters
    ----------
    df : pd.DataFrame
        The table to write.
    filename : str
        The location and name of file to
        be saved.
    bold_rows : list
        The rows to be bolded.
    """

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename+'.xls', engine='xlsxwriter')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = workbook.add_worksheet('Sheet1')

    # Add a header format.
    bold_format = workbook.add_format({'bold': True})

    # Write the column names.
    for k, val in enumerate(df.columns.values):
        worksheet.write(0, k, val)
    
    # Write the rest of the table.
    for i, row in enumerate(df.values):
        for j, val in enumerate(row):
            
            # Bold value if in the correct row.
            if i in bold_rows:
                worksheet.write(i+1, j, val, bold_format)
            
            else:
                worksheet.write(i+1, j, val)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def find_nearest(value, data):
    """Find index so that data[index] is
    closer to value than anther other.

    Parameters
    ----------
    value : float
        The value we want to find 
        an index for.
    data : 1D list
        The data to find the index
        that has the closest value.

    Returns
    -------
    index : int
        The index of data such that
        data[index] is closest to
        value.
    """

    adata = np.array(data)
    diff = np.abs(data-value)
    index = np.argmin(diff)

    return index


def organize_data(x_data, y_data):
    """Find shortests row in x_data. Get
    indices for other rows that have closest
    values.

    Parameters
    ----------
    x_data : 2D list
        Each row may have a different length. This
        data will be used to decide which indices to
        keep
    y_data : 2D list
        Each row may have a different length. The
        number of rows is expected to be the same
        as for x_data.

    Returns
    -------
    2D np.array
        Reduced versions of x_data and y_data
    """

    assert len(x_data) == len(y_data), 'x_data and y_data have different number of rows '+len(x_data)+' '+len(y_data)

    assert [len(row) for row in x_data] == [len(row) for row in y_data], 'x_data and y_data have different number of columns '+[len(row) for row in x_data] + ' ' + [len(row) for row in y_data]

    # find shortest row in x_data
    min_length_index = np.argmin([len(x) for x in x_data])

    # Go through the shortest row and find the
    # nearest value (in each other row) to the
    # elements of the shortest row.
    for i, row in enumerate(x_data):

        # Don't change anything about
        # the shortest row.
        if i == min_length_index:
            continue

        # Build a new row.
        new_row_x = []
        new_row_y = []

        # For each element of the shortest row
        for value in x_data[min_length_index]:

            print('.', end='')

            # Ge the index of the value that is closest
            index = find_nearest(value, row)

            # Put this value in the new row.
            # Use the same index for y_data
            new_row_x.append(row[index])
            new_row_y.append(y_data[i][index])

        # Apply the new row
        x_data[i] = new_row_x
        y_data[i] = new_row_y

        print()

    return np.array(x_data), np.array(y_data)


def plot_confidence_interval(x_data, y_data, color='C0', label=None):
    """Plot the mean of the columns of x_data and y_data and the
    95% confidence interval.

    Parameters
    ----------
    x_data : 2D list
        The x_data, which may not be exactly consistent.
    y_data : 2D list
        The y_data which will be averaged at each x-value.
    color : str (default='C0')
        The color to use when plotting.
    lable : str
        The legend label.

    Returns
    -------
    x : 1D np.array
        The mean of the columns of x_data after it
        was sent to organize_data.
    y : 1D np.array
        The mean of the columns of y_data after it
        was sent to organize_data.
    """

    x_data, y_data = organize_data(x_data, y_data)

    # Get the average of the columns
    x = np.mean(x_data, axis=0)
    y = np.mean(y_data, axis=0)

    # Get the confidence interval
    y_upper = np.percentile(y_data, 97.5, axis=0)
    y_lower = np.percentile(y_data, 2.5, axis=0)

    # plot
    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y_lower, y_upper, alpha=0.5, facecolor=color, edgecolor='none')

    return x, y


# x_data = [np.linspace(0, 10, 11), np.linspace(0, 10, 11)]
# y_data = [np.linspace(10, 50, 11), np.linspace(0, 40, 11)]
# print(len(y_data[0]))
# plot_confidence_interval(x_data, y_data)
# plt.show()
# exit()

# experiment number
exp = 27

# number of runs (repetitions)
nreps = 30

# significance level for mann-whitney U test
# Bonferroni correction is applied since we
# compare equation engineer to GP and to 
# semantic GP.
sig_level = 0.05/2

# Flags to include gp and semantic gp in the
# plots, or not.
genetic_programming = True
semantic_genetic_programming = False

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

function_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                  'keijzer15', 'r1', 'r2', 'r3']

# loop through the test functions
for test_function_name in function_names:        

    print('test function', test_function_name)

    if test_function_name not in ('septic'):#, 'keijzer11'):
        continue

    # This dictionary will keep track of the number of generations
    # that elapsed before the base function died off in GP.
    last_gen_given_function = {f: [] for f in function_names}

    # Get the order of training based on which function is the test function
    jumbled_target_name_indices = [(function_names.index(test_function_name)+i+1) % len(function_names)  for i, _ in enumerate(function_names)] 
    jumbled_target_names = [function_names[i] for i in jumbled_target_name_indices]

    # The data from the test function. This is the
    # data to be compared using the Mann-Whitney U test
    testdata_ee = {'error': {}, 'floating_ops': {}}
    testdata_gp = {'error': {}, 'floating_ops': {}}
    testdata_sgp = {'error': {}, 'floating_ops': {}}

    # When the training function for GP (and semantic GP)
    # changes, record the floating_ops and
    # the error at that point.
    training_switches_floating_ops_gp = []
    training_switches_floating_ops_sgp = []

    # We will plot all the data after
    # we loop through and organize it.
    # Save all the data in the following
    # lists.
    error_ee_all = []
    floating_ops_ee_all = []
    error_gp_all = []
    floating_ops_gp_all = []
    error_sgp_all = []
    floating_ops_sgp_all = []

    # for each repetition.
    for rep in range(nreps):

        print('rep', rep)

        if rep == 13:
            continue

        # First, let's get GP data
        if genetic_programming:

            # Since the GP runs are stored separately, the restart the floating op
            # count. So we need to keep track up it in this script.
            total_floating_ops_gp = 0

            # Since we need to read multiple files
            # to get this data, we will create empty
            # lists for each rep and extend them.
            error_gp_all.append([])
            floating_ops_gp_all.append([])

            # Let's loop over all the training functions.
            # By training function, I mean the function that correspond
            # to the training set.
            for training_function_name in jumbled_target_names:

                if training_function_name == test_function_name:
                    continue

                # Adjust the base path
                path_gp = os.path.join(base_data_path, 'gp', test_function_name, training_function_name)

                # Read the data
                df_gp = pd.read_csv(os.path.join(path_gp, 'best_data_rep'+str(rep)+'.csv'))

                # Since the GP runs are stored separately, the restart the floating op
                # count. So we need to keep track up it in this script.
                diff_total_floating_ops_gp = df_gp['Computation'].values[-1]

                error_gp = df_gp['Testing Error'].values

                floating_ops_gp = total_floating_ops_gp + df_gp['Computation'].values

                # We need to know when we are looking at the data
                # for the test function so that we can store the
                # testdata and get the gen on which the base
                # function (and any descendants) where killed
                # off.
                if training_function_name == test_function_name:

                    # get data for plot
                    error_gp_all[-1].extend(error_gp)
                    floating_ops_gp_all[-1].extend(floating_ops_gp)

                    # get test data
                    testdata_gp['error'][rep] = error_gp
                    testdata_gp['floating_ops'][rep] = df_gp['Computation'].values

                    # Get last gen base function was alive.
                    # This is placed in a try except block because
                    # this file is not created if the individual is
                    # never dies.
                    try:
                        with open(os.path.join(path_gp, 'generation_no_more_descendants_rep'+str(rep)+'.txt'), 'r') as f:
                            last_gen_given_function[training_function_name].append(int(f.read()))

                    except FileNotFoundError:

                        # If the individual never dies, get the last generation of training.
                        last_gen_given_function[training_function_name].append(df_gp['Generation'].values[-1])

                else:
                    
                    # get plot data
                    error_gp_all[-1].extend(error_gp)
                    floating_ops_gp_all[-1].extend(floating_ops_gp)

                    # Get more plot data. These are the points where
                    # the training dataset swiched.
                    training_switches_floating_ops_gp.append(floating_ops_gp[-1])

                # Increase the total_floating_ops, so the when the floating_ops
                # in the next run of GP starts over, we don't lose track of the
                # total count.
                total_floating_ops_gp += diff_total_floating_ops_gp

        # Now, we get the data for semantic gp (sgp).
        # TODO: record the order of target functions
        # during semantic GP.
        if semantic_genetic_programming:

            sgp_function_order = copy.copy(function_names)
            sgp_function_order.remove(test_function_name)
            sgp_function_order.append(test_function_name)

            # Like GP, the different target functions are run separately, so
            # the floating op count is reset. We will keep track of it in this
            # script.
            total_floating_ops_sgp = 0

            # Again, due to the separate target functions, we create empty lists
            # and extend them.
            error_sgp_all.append([])
            floating_ops_sgp_all.append([])
            
            # Let's loop over all the training functions.
            # By training function, I mean the function that correspond
            # to the training set.
            for training_function_name in sgp_function_order:

                path_sgp = os.path.join(base_data_path, 'semantic_gp', test_function_name)
                
                # The floating_ops and error are stored in separate files, so
                # both are opened here.
                df_sgp = pd.read_csv(os.path.join(path_sgp, 'test_error_FPM_0.0_1.0_'+training_function_name+'_rep'+str(rep)+'__'+str(exp)+'.log'))
                df_sgp_floating_ops = pd.read_csv(os.path.join(path_sgp, 'FPM_0.0_1.0_'+training_function_name+'_rep'+str(rep)+'__'+str(exp)+'.log'))

                # Keep track of total floating ops
                diff_total_floating_ops_sgp = df_sgp_floating_ops['compute'].values[-1]

                error_sgp = df_sgp['test_error'].values

                floating_ops_sgp = total_floating_ops_sgp + df_sgp_floating_ops['compute'].values


                if training_function_name == test_function_name:

                    # get plot data
                    error_sgp_all[-1].extend(error_sgp)
                    floating_ops_sgp_all[-1].extend(floating_ops_sgp)

                    # get test data, data to be compared against EE
                    # with a statistical test
                    testdata_sgp['error'][rep] = error_sgp
                    testdata_sgp['floating_ops'][rep] = df_sgp_floating_ops['compute'].values

                else:

                    # get plot data
                    error_sgp_all[-1].extend(error_sgp)
                    floating_ops_sgp_all[-1].extend(floating_ops_sgp)

                    # Get more plot data. These are the points where
                    # the training dataset swiched.
                    training_switches_floating_ops_sgp.append(floating_ops_sgp[-1])

                # update total floating ops
                total_floating_ops_sgp += diff_total_floating_ops_sgp

        # Next, let's get the data for the equation engineer
        path_ee = base_data_path

        # Equation engineer trains with multiple target functions at once, so there is no
        # switching here. Thus, no for loop. But, the training and test information is split
        # between two different files. Let's start with the training.
        df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_rep'+str(rep)+'_'+test_function_name+'.csv'))

        # Get data from training period.
        error_ee = [float(x[1:-1]) for x in df_ee['Test Error Sum'].values]
        floating_ops_ee = df_ee['Number of Floating Point Operations'].values

    #     # Now, get the testing data
    #     df_ee = pd.read_csv(os.path.join(path_ee, 'best_ind_testing_rep'+str(rep)+'_'+variant+'_'+test_function_name+'.csv'))

    #     error_ee_test = df_ee['Test Error'].values
    #     floating_ops_ee_test = df_ee['FLOPs'].values

        # Get plot data. Last point is the extension of EE to the total amound of allowed
        # floating ops even though EE was not run that long.
        # error_ee_all.append(np.hstack((error_ee, error_ee_test, [error_ee_test[-1]])))
        # floating_ops_ee_all.append(np.hstack((floating_ops_ee, floating_ops_ee_test, [2*10**10])))
        error_ee_all.append(error_ee)
        floating_ops_ee_all.append(floating_ops_ee)

        # Get test data for stat test
        testdata_ee['floating_ops'][rep] = floating_ops_ee[-1] - floating_ops_ee[0]
        testdata_ee['error'][rep] = error_ee

    # Time for the plot
    # We want to average the repetition together, however, the number of 
    # floating ops is not exactly the same between different runs, so we 
    # must get rep matrices to have consistent run length (row length, number of columns).
    # This is done with function above. So is the actual plotting. These operations can take
    # some time if there is a lot of data, so we have shorted the number of data points in GP
    # to speed this up.
    floating_ops_gp_all = [x for x in floating_ops_gp_all]
    error_gp_all = [x for x in error_gp_all]

    plt.close('all')
    plt.figure()

    # plot equation engineer data
    x, y = plot_confidence_interval(floating_ops_ee_all, error_ee_all, color='C0', label='EE')

    # indices = np.array([find_nearest(f, x) for f in np.arange(0, 10**7+1, 10**6)])
    # print(indices)
    # plt.semilogy(x[indices], y[indices], 'ok', ms=3)


    # We know that the floating ops limits are that cause the switches in first place, so
    # these should be very close. They will be slightly different because of the operations
    # done before checking.
    training_switches_floating_ops_gp = training_switches_floating_ops_sgp = [i*5*10**9 for i in range(1, 11)]

    if genetic_programming:

        # skips = 1

        # floating_ops_gp_all = [x[::skips] for x in floating_ops_gp_all]
        # error_gp_all = [x[::skips] for x in error_gp_all]

        floating_ops_gp_all = [x for x in floating_ops_gp_all]
        error_gp_all = [x for x in error_gp_all]

        # This time we want the averages so that we can plot the training switches. I don't want
        # to draw vertical lines because equation engineer does not switch training. Maybe I should
        # have it switch, so it is more similar? Start with one target and add one at every switch?
        x, y = plot_confidence_interval(floating_ops_gp_all, error_gp_all, color='C1', label='GP')

        # Get the nearest floating ops and get the indices, to be able to plot corresponding
        # averaged error.
        # indices = np.array([find_nearest(f, x) for f in training_switches_floating_ops_gp])

        # # plot it
        # plt.semilogy(x[indices], y[indices], 'ok', ms=3)

        print('train points plotted')

    if semantic_genetic_programming:

        # Now, we do the same thing for semantic gp
        x, y = plot_confidence_interval(floating_ops_sgp_all, error_sgp_all, color='C2', label='Semantic GP')

        indices = np.array([find_nearest(f, x) for f in training_switches_floating_ops_sgp])

        plt.semilogy(x[indices], y[indices], 'ok', ms=3)

    switches = np.arange(0, 10**8+1, 10**7)
    x = []
    y = []
    ylim = plt.ylim()
    # for s in switches:

    #     plt.semilogy([s]*2, ylim, '--k', zorder=0, label='Training Switch')

    # Remove duplicat labels in the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Additional plot details
    plt.title(test_function_name)
    plt.xlabel('Number of Floating Point Operations')
    plt.ylabel('Test Error')

    plt.savefig(os.path.join(save_path, 'ee_gp_floating_ops_'+test_function_name+'.pdf'))

    # Time for stats: compare (p-value) test values at max floating_ops for EE
    ee_test_final = [testdata_ee['error'][rep][-1] for rep in testdata_ee['error']]

    # Take the max number of floating ops during testing of EE to use for the comparison.
    floating_ops_level = np.max([testdata_ee['floating_ops'][rep] for rep in testdata_ee['floating_ops']])

    print('floating_ops_level', floating_ops_level)
    
    # This if statement, is here so commenting out
    # the gp data collection does not break the script.
    if testdata_gp['error']:

        # rep: index
        gp_indices = {}

        # Get the index corresponding to the first floating op count
        # that is greater than the one for EE. There error at this point
        # will be used for the comparison.
        for rep in testdata_gp['floating_ops']:
            for i, c in enumerate(testdata_gp['floating_ops'][rep]):
                if c > floating_ops_level:
                    gp_indices[rep] = i
                    break

        # Use the indices to get the error
        gp_test_final = [testdata_gp['error'][rep][gp_indices[rep]] for rep in gp_indices]

        # Do the test: Is EE < GP in terms of test error for given amount of floating ops?
        stat, pvalue = scipy.stats.mannwhitneyu(ee_test_final, gp_test_final, alternative='less')
        print(test_function_name, 'ee < gp', pvalue)

        # Update p-value table
        table = table.append({'Target': test_function_name, 'Test': 'ee < gp', 'p-value': pvalue, 'Significance Level': sig_level}, ignore_index=True)

        # If the p-value is significant,
        # bold the row.
        if pvalue < sig_level:

            current_row = table.shape[0] - 1
            bold_rows.append(current_row)


    # This if statement, is here so commenting out
    # the semantic gp data collection does not break the script.
    if testdata_sgp['error']:

        sgp_indices = {}

        # Get the index corresponding to the first floating op count
        # that is greater than the one for EE. There error at this point
        # will be used for the comparison.
        for rep in testdata_sgp['floating_ops']:
            for i, c in enumerate(testdata_sgp['floating_ops'][rep]):
                if c > floating_ops_level:
                    sgp_indices[rep] = i
                    break

        # Use the indices to get the error
        sgp_test_final = [testdata_sgp['error'][rep][sgp_indices[rep]] for rep in sgp_indices]

        # Do the test: Is EE < semantic GP in terms of test error for given amount of floating ops?
        stat, pvalue = scipy.stats.mannwhitneyu(ee_test_final, sgp_test_final, alternative='less')
        print(test_function_name, 'ee < semantic_gp', pvalue)

        # update p-value table
        table = table.append({'Target': test_function_name, 'Test': 'ee < semantic gp', 'p-value': pvalue, 'Significance Level': sig_level}, ignore_index=True)

        # If the p-value is significant,
        # bold the row.
        if pvalue < sig_level:

            current_row = table.shape[0] - 1
            bold_rows.append(current_row)

    # New plot of the data actually being compared.
    plt.close('all')
    plt.figure()
    plot_emprical_cumulative_distribution_funtion(ee_test_final, label='ee')

    # This if statement, is here so commenting out
    # the gp data collection does not break the script.
    if testdata_gp['error']:
        plot_emprical_cumulative_distribution_funtion(gp_test_final, label='gp')

    # This if statement, is here so commenting out
    # the semantic gp data collection does not break the script.
    if testdata_sgp['error']:
        plot_emprical_cumulative_distribution_funtion(sgp_test_final, label='semantic gp')

    plt.xscale('log')
    plt.xlabel('Final Test Error')
    plt.title(test_function_name)

    # Remove duplicat labels in the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(os.path.join(save_path, 'ee_gp_floating_ops_cdf_'+test_function_name+'.pdf'))

# save the table
write_table_with_bold_rows(df=table, filename=os.path.join(stats_save_path, 'stats_ee_gp_floating_ops_exp'+str(exp)), bold_rows=bold_rows)

# Next plot: last generation that base function exists in GP runs.
plt.figure()

# For all the target functions. Unlike the other figures,
# this one has the data for each test target function on
# the same plot.
for i, f in enumerate(last_gen_given_function):

    # Since there are only 10 default colors and 11 target functions,
    # make the zero-th one black.
    if i > 0:
        plot_emprical_cumulative_distribution_funtion(last_gen_given_function[f], label=f)

    else:
        plot_emprical_cumulative_distribution_funtion(last_gen_given_function[f], label=f, color='k')


plt.xlabel('Last Generation Descenant of Given Function is Alive')
plt.xscale('log')
plt.legend()

plt.savefig(os.path.join(save_path, 'last_gen_alive.pdf'))
