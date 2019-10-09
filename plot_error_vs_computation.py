import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

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

exp = 5
nreps = 29
data_ea = {'test error': {}, 'computation': {}}
data_gp = {'test error': {}, 'computation': {}}
data_sgp = {'test error': {}, 'computation': {}}

# # pick best ea
# val_avg = []
# xopts = []

# for rep in range(nreps):

#     path_ea = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))
#     path_gp = os.path.join(path_ea, 'gp')

#     df_ea = pd.read_csv(os.path.join(path_ea, 'best_ind_validation_rep'+str(rep)+'.csv'))

#     val_avg.append(np.mean(df_ea['Validation Fitness']))

#     xopts.append(pd.read_csv(os.path.join(path_ea, 'best_ind_rep'+str(rep)+'.csv')).iloc[:,1:].values)


# # get rep the produces best ea
# rep_index = np.argmin(val_avg)

# print('val_avg', val_avg)
# print('rep_index', rep_index)

# import equation_correction as ea

# # should be able to get all this from summary
# horizontal = False
# num_adjustments = 50

# EA = ea.EquationAdjustor(initial_hidden_values=np.random.uniform(-1, 1, size=10),
#                          hidden_weights=np.random.uniform(-1, 1, size=(10,10)), activation=np.tanh, horizontal=horizontal,
#                          initial_adjustment=1, initial_parameter=0, num_adjustments=50, fixed_adjustments=False)

# fake_dataset = np.array([[1,2], [1,2]])

# if not horizontal:
#     max_shift = sum([1-k/num_adjustments for k in range(num_adjustments)])

# else:
#     max_shift = 5

# bench_errors = []

# for rep in range(nreps):

#     benchmark_datasets = ea.get_benchmark_datasets(np.random.RandomState(rep+100*exp), max_shift, horizontal)

#     all_datasets = {'training': [['x[0]', fake_dataset, fake_dataset, fake_dataset]], 'validation': [['x[0]', fake_dataset, fake_dataset, fake_dataset]], 'testing': benchmark_datasets}

#     errors, fitnesses = EA.cma_es_function(xopts[rep_index].flatten(), np.random.RandomState(0), all_datasets, return_all_errors=True)

#     bench_errors.append(errors['testing'])

# exit()

for function_name in ['quartic']:

    for rep in range(nreps):

        path_ea = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))
        path_gp = os.path.join(path_ea, 'gp')
        path_sgp = os.path.join(path_ea, 'semantic_gp')

        df_ea = pd.read_csv(os.path.join(path_ea, 'best_ind_testing_rep'+str(rep)+'.csv'))
        df_gp = pd.read_csv(os.path.join(path_gp, function_name, 'best_data_rep'+str(rep)+'.csv'))
        df_sgp = pd.read_csv(os.path.join(path_sgp, 'FPM_0.0_1.0_'+function_name+'_'+str(rep+100*exp+1)+'.log'))

        error_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Test Error'].values if is_number(x)])
        # error_ea = error_ea[~np.isnan(error_ea)]
        computation_ea = np.array([float(x) for x in df_ea.loc[df_ea['Target'] == function_name]['Computation'].values if is_number(x)])
        # computation_ea = computation_ea[~np.isnan(computation_ea)]

        data_ea['test error'][rep] = error_ea
        data_ea['computation'][rep] = computation_ea

        error_gp = np.array([float(x) for x in df_gp['Testing Error'].values if is_number(x)])
        computation_gp = np.array([float(x) for x in df_gp['Computation'].values if is_number(x)])

        data_gp['test error'][rep] = error_gp
        data_gp['computation'][rep] = computation_gp

        with open(os.path.join(path_sgp, 'cycles_per_second_'+str(rep+100*exp+1)+'.txt'), 'r') as f:
            cycles_per_second = float(f.read())

        error_sgp = df_sgp['min_fitness'].values
        computation_sgp = cycles_per_second*df_sgp['cpu_time'].values

        data_sgp['test error'][rep] = error_sgp
        data_sgp['computation'][rep] = computation_sgp

        error_gp, computation_gp = remove_duplicates(error_gp, computation_gp)
        error_sgp, computation_sgp = remove_duplicates(error_sgp, computation_sgp)

        # plt.close('all')
        # plt.figure()
        plt.semilogy(computation_ea, error_ea+10**(-16), '-', color='C0', label='ea')
        plt.plot(computation_gp, error_gp+10**(-16), '-', color='C1', label='gp')
        plt.plot(computation_sgp, error_sgp+10**(-16), '-', color='C2', label='semantic_gp')

        plt.xlabel('Computation = (CPU Time) $\\times$ (cycles per second)')
        plt.ylabel('Test Error + $10^{-16}$')
    plt.legend(['ea', 'gp'])
    # plt.show()
    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp), 'figures', 'ea_gp_comp_'+function_name+'.pdf'))

    # compare (p-value) test values at max computation for EA
    ea_test_final = [data_ea['test error'][rep][-1] for rep in data_ea['test error']]

    computation_level = np.max([data_ea['computation'][rep] for rep in data_ea['computation']])
    print(computation_level)

    # rep: index
    gp_indices = {}

    for rep in data_gp['computation']:
        for i, c in enumerate(data_gp['computation'][rep]):
            if c > computation_level:
                gp_indices[rep] = i
                break


    gp_test_final = [data_gp['test error'][rep][gp_indices[rep]] for rep in gp_indices]

    results = scipy.stats.mannwhitneyu(ea_test_final, gp_test_final, alternative='less')
    print(function_name, 'H0: ea < gp', results)

    sgp_indices = {}

    for rep in data_sgp['computation']:
        for i, c in enumerate(data_sgp['computation'][rep]):
            if c > computation_level:
                sgp_indices[rep] = i
                break

    sgp_test_final = [data_sgp['test error'][rep][sgp_indices[rep]] for rep in sgp_indices]

    results = scipy.stats.mannwhitneyu(ea_test_final, sgp_test_final, alternative='less')
    print(function_name, 'H0: ea < semantic_gp', results)

