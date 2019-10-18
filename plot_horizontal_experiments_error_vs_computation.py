import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from collections import OrderedDict
import os
import copy

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

exp = 16
nreps = 30

save_path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures')

if not os.path.exists(save_path):
    os.makedirs(save_path)

experiments = [ 'MO1_RO0_PE0_ED0_TS0_HL2',
                'MO1_RO0_PE1_ED0_TS0_HL2',
                'MO1_RO1_PE1_ED0_TS0_HL2',
                'MO0_RO1_PE1_ED0_TS0_HL2',
                'MO0_RO1_PE0_ED1_TS0_HL2',
                'MO0_RO0_PE0_ED1_TS0_HL2',
                'MO0_RO0_PE1_ED0_TS0_HL2']

data = {}

for variant in experiments:
    data[variant] = {'test error': {}, 'computation': {}}

# data_ea = {'test error': {}, 'computation': {}}
# data_gp = {'test error': {}, 'computation': {}}
# data_sgp = {'test error': {}, 'computation': {}}

boxplot_data = []

for function_name in ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                      'keijzer15', 'r1', 'r2', 'r3']:

    boxplot_data = []

    for i, variant in enumerate(data):

        if len(boxplot_data) < len(data):
            boxplot_data.append([])

        plt.close('all')
        plt.figure()

        for rep in range(nreps):

            path_ea = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))

            df_ea = pd.read_csv(os.path.join(path_ea, 'best_ind_testing_rep'+str(rep)+'_'+variant+'.csv'))
            error_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Test Error'].values if is_number(x)])
            # error_ea = error_ea[~np.isnan(error_ea)]
            computation_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Computation'].values if is_number(x)])
            # computation_ea = computation_ea[~np.isnan(computation_ea)]

            data[variant]['test error'][rep] = error_ea
            data[variant]['computation'][rep] = computation_ea

            boxplot_data[i].append(error_ea[-1])

            # plt.close('all')
            plt.semilogy(computation_ea, error_ea+10**(-16), '-', color='C'+str(i), label=variant)

            plt.xlabel('Computation = (CPU Time) $\\times$ (cycles per second)')
            plt.ylabel('Test Error + $10^{-16}$')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # plt.show()
        plt.title(function_name)
        plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_gp_comp_'+function_name+'.pdf'))

    plt.figure()
    plt.boxplot(boxplot_data)
    plt.yscale('log')
    plt.xticks(list(range(len(data))), list(data.keys()), rotation=45)
    plt.ylabel('Testing Errror')
    plt.title(function_name)
    plt.tight_layout()
    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_boxplots_'+function_name+'.pdf'))

    print(function_name)

    index = 2

    for i, b in enumerate(boxplot_data):

        if index == i:
            continue

        stat, pvalue = scipy.stats.mannwhitneyu(boxplot_data[index], b, alternative='less')

        print(experiments[index], '<', experiments[i], pvalue*(len(experiments)-1))

    print('')

        # compare (p-value) test values at max computation for EA
        # ea_test_final = [data_ea['test error'][rep][-1] for rep in data_ea['test error']]

        # computation_level = np.max([data_ea['computation'][rep] for rep in data_ea['computation']])
        # print(computation_level)

        # # rep: index
        # gp_indices = {}

        # for rep in data_gp['computation']:
        #     for i, c in enumerate(data_gp['computation'][rep]):
        #         if c > computation_level:
        #             gp_indices[rep] = i
        #             break


        # gp_test_final = [data_gp['test error'][rep][gp_indices[rep]] for rep in gp_indices]

        # results = scipy.stats.mannwhitneyu(ea_test_final, gp_test_final, alternative='less')
        # print(function_name, 'H0: ea < gp', results)

        # sgp_indices = {}

        # for rep in data_sgp['computation']:
        #     for i, c in enumerate(data_sgp['computation'][rep]):
        #         if c > computation_level:
        #             sgp_indices[rep] = i
        #             break

        # sgp_test_final = [data_sgp['test error'][rep][sgp_indices[rep]] for rep in sgp_indices]

        # results = scipy.stats.mannwhitneyu(ea_test_final, sgp_test_final, alternative='less')
        # print(function_name, 'H0: ea < semantic_gp', results)

