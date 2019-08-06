# Overwrite (or create) the constants and protected functions
import pickling_setup.protected_functions_writer_fb
import pickling_setup.consts_writer_fb

# Read in the constants and protected functions
from GeneticProgramming.protected_functions import *
from GeneticProgramming.consts import *

# Now import like normal
from GeneticProgramming.GeneticProgrammingAfpo import GeneticProgrammingAfpo
from GeneticProgramming.GeneticProgrammingRestricted import GeneticProgrammingRestricted
from GeneticProgramming.IndividualRestricted import IndividualRestricted
import GeneticProgramming.data_setup as ds
import GeneticProgramming.common_functions as cf

from get_computation import get_computation_time

import numpy as np
import pandas as pd
from interval import interval

import os
import argparse
import time
import collections

# --------------------------------------------------------- #
#                      PARAMETERS
# --------------------------------------------------------- #

parser = argparse.ArgumentParser()

# this will act as an index for rep_list (offset by 1 though)
parser.add_argument('rep', help='Number of runs already performed', type=int)

parser.add_argument('func', help='Specify the target function', type=str)
parser.add_argument('exp', help='exp is the experiment number', type=int)

# parser.add_argument('-s', '--size', help='Use tree size as second objective',
#                     action='store_true')
# parser.add_argument('-uia', '--use_interval_arithmetic', help='Use interval arithmetic'
#                     'to check if solutions have'
#                     'undefined output',
#                     action='store_true')
# parser.add_argument('-afspo', '--AFSPO', help='Use the number of nodes in tree as third objective',
#                     action='store_true')
parser.add_argument('-re', '--redos', help='Specific runs to do',
                    type=str, action='store', default='')

parser.add_argument('-t', '--timeout', help='Number of seconds after which to stop.',
                    action='store', type=float, default=108000.)

parser.add_argument('-r', '--restriction', help='If False, run standard GP instead of restricted.',
                    action='store_false')

args = parser.parse_args()

print(args)

key = args.func    # target function to use
run_index = args.rep

exp = args.exp

mutation_param = 2    # not going to change this

nreps = 100

if args.redos == '':

    doing_redos = False
    run_list = list(range(nreps))

else:

    doing_redos = True
    run_list = list(map(int, args.redos.split(',')))
    print(run_list)

noise_std = 0.1

# --------------------------------------------------------- #
#                      END PARAMETERS
# --------------------------------------------------------- #

def get_partial_fills(primitives, num_fills):
    """Get dictionary that will be used to fill the tree.

    Parameters
    ----------
    num_non_leaf : int
        The number of nodes that are not leaves (and not root).
        These are the nodes where primitives are placed.
    num_labels : int
        The number of primitives (or possible labels).
    num_fills : int
        The number of partial fills to create.

    Returns
    -------
    pfills : list
        A list of dictionaries describing the partial fill
        where key=location and value=label.
    """

    # k is the number of non-leaf nodes to fill.
    # m = len(primitives)
    # n = len(locations)
    # p = [m**(k-1)*(m-1)/(m**(n)-1) for k in range(1, n+1)]

    pfills = [{(): p} for p in primitives]

    # # This will happen if depth = 1
    # if n == 0:
    #     return pfills

    # terminal_depth = max(map(len, locations))

    # # Give computed constants a number (like c0)
    # # and keep track of the number to avoid unnecessary
    # # duplicate. There are only 10 constants so if
    # # consts_count gets to 10 repeats will begin.
    # const_count = 0

    # # Never pick a non-computed constant. We don't
    # # want to be guessing the value of the const.
    # sub_terminals = copy.copy(terminals)

    # if '#f' in sub_terminals:
    #     sub_terminals.remove('#f')

    # for _ in range(num_fills-len(primitives)):

    #     k = rng.choice(range(1, n))

    #     # Do this one uniformly and k times
    #     indices = rng.choice(n, size=k, replace=False)
    #     locs_subset = [locations[i] for i in indices]
    #     locs_labels = []

    #     for loc in locs_subset:

    #         if len(loc) == terminal_depth:

    #             # Never pick a non-computed constant. We don't
    #             # want to be guessing the value of the const.
    #             locs_labels.append(rng.choice(sub_terminals))

    #             # Pick an actual constant (for computed consts)
    #             # if that is what was picked.
    #             if locs_labels[-1] == '#c':

    #                 locs_labels[-1] = 'c%i' % (const_count % 10,)
    #                 const_count += 1

    #         else:
    #             locs_labels.append(rng.choice(primitives))


    #     pfill = {(): rng.choice(primitives)}

    #     for loc, label in zip(locs_subset, locs_labels):
    #         pfill[loc] = label

    #     pfills.append(pfill)

    return pfills


# initialize and run gp
primitive_set = ['+', '*', '-', '%', 'sin2', 'id2']
terminal_set = ['#x', '#f']

restrictions = get_partial_fills(primitive_set, 6)


def run_single(rng, pop_size, primitive_set, terminal_set,
               prob_mutate, prob_xover, mutation_param,
               rep, output_path, output_file, **params):

    if args.func == 'test':

        num_data_points = 200
        frac_train = 0.5

        target = lambda x: 2*x[0]

        X1 = np.linspace(-1, 1, num_data_points)
        X2 = np.linspace(-20, 3, num_data_points)
        rng.shuffle(X2)
        X = np.vstack((X1, X2))

        dataset = np.vstack((target(X), X1, X2)).T

        indices_train = rng.choice(num_data_points, size=int(num_data_points*frac_train), replace=False)
        indices_test = np.array([i for i in range(num_data_points) if i not in indices_train])
        dataset_train = dataset[indices_train]
        test_data = dataset[indices_test]

        dataset = np.array([dataset_train, dataset_train])

        num_vars = 2

    # if target function
    elif 'f' in function_dict[key]:
        dataset = ds.get_datasets(rng=np.random.RandomState(rep),
                                  f=function_dict[key]['f'],
                                  A=function_dict[key]['a'],
                                  B=function_dict[key]['b'],
                                  noise_percent=None,
                                  noise_std=noise_std,
                                  data_size=function_dict[key]['size'])

        num_vars = len(dataset[0][0])-1
        print('num_vars', num_vars)

        # always use the same seed for each run in exp
        test_data = ds.get_datasets(rng=np.random.RandomState(exp),
                                    f=function_dict[key]['f'],
                                    A=function_dict[key]['a'],
                                    B=function_dict[key]['b'],
                                    noise_percent=None,
                                    noise_std=noise_std,
                                    data_size=int(100000 / num_vars))[0]

        A = function_dict[key]['a']
        B = function_dict[key]['b']

        if type(A) in (float, int):

            A = [A]
            B = [B]

        params['interval'] = [interval([a, b]) for a, b in zip(A, B)]

    # if dataset
    elif 'path' in function_dict[key]:
        path = os.path.join(os.environ['DATASET_PATH'],
                            function_dict[key]['path'], 'data.csv')

        data = pd.read_csv(path).iloc[:, :].values

        dataset, test_data = ds.split_data(np.random.RandomState(rep), data, (1, 1, 5))
        print(dataset)

        left_endpoints = [np.min(x) for x in dataset[:, 1:].T]
        right_endpoints = [np.max(x) for x in dataset[:, 1:].T]
        input_endpoints = np.vstack((left_endpoints, right_endpoints)).T
        params['interval'] = [interval([a, b]) for a, b in input_endpoints]
        print(params['interval'])

        num_vars = len(dataset[0][0])-1
        print('num_vars', num_vars)

    if args.restriction:
        gp = GeneticProgrammingRestricted(rng=rng,
                                          pop_size=pop_size,
                                          primitive_set=primitive_set,
                                          terminal_set=terminal_set,
                                          # this is not data, which is passed
                                          data=dataset,
                                          test_data=test_data,
                                          prob_mutate=prob_mutate,
                                          prob_xover=prob_xover,
                                          num_vars=num_vars,
                                          mutation_param=mutation_param,
                                          individual=IndividualRestricted,
                                          # parameters below
                                          **params)

    else:
        gp = GeneticProgrammingAfpo(rng=rng,
                                    pop_size=pop_size,
                                    primitive_set=primitive_set,
                                    terminal_set=terminal_set,
                                    # this is not data, which is passed
                                    data=dataset,
                                    test_data=test_data,
                                    prob_mutate=prob_mutate,
                                    prob_xover=prob_xover,
                                    num_vars=num_vars,
                                    mutation_param=mutation_param,
                                    # parameters below
                                    **params)

    info = gp.run(rep=rep,
                  output_path=output_path,
                  output_file=output_file)

    return info


params = collections.OrderedDict()
params['T'] = args.timeout
params['R'] = args.restriction

last_folder = cf.get_folder(params) + '/'

params['IA'] = False
params['AFSPO'] = False
params['size'] = False
params['count_asymptotes'] = False
params['save_pop_data'] = False
print(params['T'])
# Adjust time
params['T'] = get_computation_time(params['T'])
print(params['T'])
exit()
print('last_folder', last_folder)

if len(run_list) <= run_index:

    print('WARNING: run_index out of range. If redos, this is expected.')
    exit()

start = time.time()

params['T'] /= len(restrictions)

for i, res in enumerate(restrictions):

    params['restrictions'] = res

    run_single(rng=np.random.RandomState(run_list[run_index] + exp), pop_size=population_size,
               primitive_set=primitive_set,
               terminal_set=terminal_set, prob_mutate=1.,
               prob_xover=0.,
               mutation_param=mutation_param, rep=run_list[run_index],
               output_path=os.path.join(os.environ['GP_DATA'], 'function_builder_comparison/experiments/' + str(exp) + '/' + key + '/restriction'+str(i)+'/'+last_folder),
               output_file='fitness_data_rep' + str(run_list[run_index]) + '.csv', **params)

print(time.time() - start)
