import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import os

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


exp = 5
data_ea = {'test error': {}, 'computation': {}}
data_gp = {'test error': {}, 'computation': {}}

for rep in range(10):

    path_ea = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))
    path_gp = os.path.join(path_ea, 'gp')

    with open(os.path.join(path_ea, 'test_computation_and_cycles_rep'+str(rep)+'.txt'), mode='r') as f:
        cycles_per_second_ea = float(f.read().split(' ')[1])

    df_ea = pd.read_csv(os.path.join(path_ea, 'best_ind_rep'+str(rep)+'.csv'))
    df_gp = pd.read_csv(os.path.join(path_gp, 'best_data_rep'+str(rep)+'.csv'))

    error_ea = np.array([float(x) for x in df_ea['test error on test function'].values if is_number(x)])
    error_ea = error_ea[~np.isnan(error_ea)]
    computation_ea = cycles_per_second_ea*np.array([0] + [float(x) for x in df_ea['cpu time'].values if is_number(x)])
    computation_ea = computation_ea[~np.isnan(computation_ea)]

    data_ea['test error'][rep] = error_ea
    data_ea['computation'][rep] = computation_ea

    error_gp = np.array([float(x) for x in df_gp['Testing Error'].values if is_number(x)])
    computation_gp = np.array([float(x) for x in df_gp['Computation'].values if is_number(x)])

    data_gp['test error'][rep] = error_gp
    data_gp['computation'][rep] = computation_gp

    error_gp, computation_gp = remove_duplicates(error_gp, computation_gp)

    # plt.close('all')
    # plt.figure()
    plt.semilogy(computation_ea, error_ea+10**(-16), '-', color='C0', label='ea')
    plt.plot(computation_gp, error_gp+10**(-16), '-', color='C1', label='gp')

    plt.xlabel('Computation = (CPU Time) $\\times$ (cycles per second)')
    plt.ylabel('Test Error + $10^{-16}$')
plt.legend(['ea', 'gp'])
# plt.show()
plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_gp_comp.pdf'))

# compare (p-value) test values at max computation for EA
ea_test_final = [data_ea['test error'][rep][-1] for rep in data_ea['test error']]

computation_level = np.max([data_ea['computation'][rep] for rep in data_ea['computation']])
print(computation_level)

# rep: index
indices = {}

for rep in data_gp['computation']:
    for i, c in enumerate(data_gp['computation'][rep]):
        if c > computation_level:
            indices[rep] = i

gp_test_final = [data_gp['test error'][rep][indices[rep]] for rep in indices]

results = scipy.stats.mannwhitneyu(ea_test_final, gp_test_final, alternative='less')
print(results)

