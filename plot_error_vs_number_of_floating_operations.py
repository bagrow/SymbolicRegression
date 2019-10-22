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
        float(x)
        return True

    except ValueError:
        return False


def remove_duplicates(x1, x2):
    """Take two equal length lists where x1 has many duplicate values
    find the first instance of a duplicate and the last.

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

    indices = []
    prev = None
    duplicates = False

    for i, x in enumerate(x1):

        if prev is None:
            indices.append(i)

        else:

            if x == prev:
                duplicates = True
                continue

            else:

                if duplicates:
                    indices.append(i-1)
                    duplicates = False

                indices.append(i)

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

    p = [(i)/n for i, x in enumerate(X)]

    return p, X_sorted


def plot_emprical_cumulative_distribution_funtion(X, labels=True, label=None):

    p, X = get_emprical_cumulative_distribution_funtion(X)

    plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4, label=label)

    if labels:

        plt.yticks(np.linspace(0, 1, 6))
        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')


def write_table_with_bold_rows(df, filename, bold_rows):

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename+'.xls', engine='xlsxwriter')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = workbook.add_worksheet('Sheet1')

    # Add a header format.
    bold_format = workbook.add_format({'bold': True})

    for k, val in enumerate(df.columns.values):
        worksheet.write(0, k, val)
    
    for i, row in enumerate(df.values):
        for j, val in enumerate(row):
            
            if i in bold_rows:
                worksheet.write(i+1, j, val, bold_format)
            
            else:
                worksheet.write(i+1, j, val)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


exp = 18
nreps = 30
sig_level = 0.05/2

save_path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures')
stats_save_path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'stats')

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(stats_save_path):
    os.makedirs(stats_save_path)

table = pd.DataFrame([], columns=['Target', 'Test', 'p-value', 'Significance Level'])
bold_rows = []

function_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                  'keijzer15', 'r1', 'r2', 'r3']

last_gen_given_function = {f: [] for f in function_names}

for test_function_name in function_names:

    jumbled_target_name_indices = [(function_names.index(test_function_name)+i+1) % len(function_names)  for i, _ in enumerate(function_names)] 
    jumbled_target_names = [function_names[i] for i in jumbled_target_name_indices]

    plt.close('all')
    plt.figure()

    data_ea = {'test error': {}, 'computation': {}}
    data_gp = {'test error': {}, 'computation': {}}
    data_sgp = {'test error': {}, 'computation': {}}

    training_switches_comp = []
    training_switches_error = []

    for rep in range(nreps):

        # rep = 1
        print('rep', rep)

        path_ea = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))

        total_computation = 0

        for training_function_name in jumbled_target_names:

            path_gp = os.path.join(path_ea, 'gp', test_function_name, training_function_name)
            # path_sgp = os.path.join(path_ea, 'semantic_gp')

            df_gp = pd.read_csv(os.path.join(path_gp, 'best_data_rep'+str(rep)+'.csv'))

            diff_total_computation = df_gp['Computation'].values[-1]

            error_gp = df_gp['Testing Error'].values/df_gp['Testing Error'].values[0]

            computation_gp = total_computation + df_gp['Computation'].values

            total_computation += diff_total_computation

            if training_function_name == test_function_name:
                plt.semilogy(computation_gp, error_gp, '-', color='C1', label='gp')


            else:

                plt.semilogy(computation_gp, error_gp, '-', color='C1', label='gp', alpha=0.5)

                training_switches_error.append(error_gp[-1])
                training_switches_comp.append(computation_gp[-1])
            # plt.text(np.mean(computation_gp), np.max(error_gp), training_function_name)

            try:
                with open(os.path.join(path_gp, 'generation_no_more_descendants_rep'+str(rep)+'.txt'), 'r') as f:
                    last_gen_given_function[training_function_name].append(int(f.read()))

            except FileNotFoundError:
                last_gen_given_function[training_function_name].append(df_gp['Generation'].values[-1])

        # df_ea = pd.read_csv(os.path.join(path_ea, 'best_ind_testing_rep'+str(rep)+'_MO1_RO1_PE1_ED0.csv'))
        # df_sgp_test = pd.read_csv(os.path.join(path_sgp, 'test_error_FPM_0.0_1.0_'+function_name+'_'+str(rep+100*exp)+'.log'))
        # df_sgp_cpu = pd.read_csv(os.path.join(path_sgp, 'FPM_0.0_1.0_'+function_name+'_'+str(rep+100*exp)+'.log'))

        # error_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Test Error'].values if is_number(x)])
        # computation_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Computation'].values if is_number(x)])

        # data_ea['test error'][rep] = error_ea
        # data_ea['computation'][rep] = computation_ea

        # with open(os.path.join(path_sgp, 'cycles_per_second_exp'+str(exp)+'_rep'+str(rep)+'.txt'), 'r') as f:
        #     cycles_per_second = float(f.read())

        # error_sgp = df_sgp_test['test_error'].values
        # computation_sgp = cycles_per_second*df_sgp_cpu['cpu_time'].values

        # data_sgp['test error'][rep] = error_sgp
        # data_sgp['computation'][rep] = computation_sgp

        # error_gp, computation_gp = remove_duplicates(error_gp, computation_gp)
        # error_sgp, computation_sgp = remove_duplicates(error_sgp, computation_sgp)

        # plt.semilogy(computation_sgp, error_sgp, '-', color='C2', label='semantic_gp')
        # plt.semilogy(computation_ea, error_ea, '-', color='C0', label='ea')

        plt.xlabel('Number of Floating Point Computations')
        plt.ylabel('Test Error')

    plt.semilogy(training_switches_comp, training_switches_error, 'ok', ms=3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(test_function_name)

    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_gp_comp_'+test_function_name+'.pdf'))
    
    plt.figure()
    for f in last_gen_given_function:
        plot_emprical_cumulative_distribution_funtion(last_gen_given_function[f], label=f)

    plt.xlabel('Last Generation Descenant of Given Function is Alive')
    plt.legend()
    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'last_gen_alive_'+test_function_name+'.pdf'))
    exit()
    # compare (p-value) test values at max computation for EA
    ea_test_final = [data_ea['test error'][rep][-1] for rep in data_ea['test error']]

    computation_level = np.max([data_ea['computation'][rep] for rep in data_ea['computation']])

    # rep: index
    gp_indices = {}

    for rep in data_gp['computation']:
        for i, c in enumerate(data_gp['computation'][rep]):
            if c > computation_level:
                gp_indices[rep] = i
                break


    gp_test_final = [data_gp['test error'][rep][gp_indices[rep]] for rep in gp_indices]

    stat, pvalue = scipy.stats.mannwhitneyu(ea_test_final, gp_test_final, alternative='less')
    print(function_name, 'ea < gp', pvalue)

    table = table.append({'Target': function_name, 'Test': 'ea < gp', 'p-value': pvalue, 'Significance Level': sig_level}, ignore_index=True)

    if pvalue < sig_level:

        current_row = table.shape[0] - 1
        bold_rows.append(current_row)

    sgp_indices = {}

    for rep in data_sgp['computation']:
        for i, c in enumerate(data_sgp['computation'][rep]):
            if c > computation_level:
                sgp_indices[rep] = i
                break

    sgp_test_final = [data_sgp['test error'][rep][sgp_indices[rep]] for rep in sgp_indices]

    stat, pvalue = scipy.stats.mannwhitneyu(ea_test_final, sgp_test_final, alternative='less')
    print(function_name, 'ea < semantic_gp', pvalue)

    table = table.append({'Target': function_name, 'Test': 'ea < semantic gp', 'p-value': pvalue, 'Significance Level': sig_level}, ignore_index=True)

    if pvalue < sig_level:

        current_row = table.shape[0] - 1
        bold_rows.append(current_row)

    plt.close('all')
    plt.figure()
    plot_emprical_cumulative_distribution_funtion(ea_test_final, label='ea')
    plot_emprical_cumulative_distribution_funtion(gp_test_final, label='gp')
    plot_emprical_cumulative_distribution_funtion(sgp_test_final, label='semantic gp')
    plt.xscale('log')
    plt.xlabel('Final Test Error')
    plt.title(function_name)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_gp_comp_cdf_'+function_name+'.pdf'))

write_table_with_bold_rows(df=table, filename=os.path.join(stats_save_path, 'stats_ea_gp_comp_exp'+str(exp)), bold_rows=bold_rows)
