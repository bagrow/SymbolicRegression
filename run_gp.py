# Overwrite (or create) the constants and protected functions
import protected_functions_writer
import consts_writer

# Read in the constants and protected functions
from GeneticProgramming.protected_functions import *
from GeneticProgramming.consts import *

# Now import like normal
from GeneticProgramming.GeneticProgrammingAfpo import GeneticProgrammingAfpo
from GeneticProgramming.GeneticProgramming import GeneticProgramming
import GeneticProgramming.data_setup as ds
import GeneticProgramming.common_functions as cf

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

parser.add_argument('-s', '--size', help='Use tree size as second objective',
                    action='store_true')
parser.add_argument('-uia', '--use_interval_arithmetic', help='Use interval arithmetic'
                    'to check if solutions have'
                    'undefined output',
                    action='store_true')
parser.add_argument('-afspo', '--AFSPO', help='Use the number of nodes in tree as third objective',
                    action='store_true')
parser.add_argument('-re', '--redos', help='Specific runs to do',
                    type=str, action='store', default='')

args = parser.parse_args()

print(args)

key = args.func    # target function to use
run_index = args.rep - 1

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

# initialize and run gp
primitive_set = ['+', '*', '-', '%']
terminal_set = ['#x', '#f']


def run_single(rng, pop_size, primitive_set, terminal_set,
               prob_mutate, prob_xover, mutation_param,
               rep, output_path, output_file, **params):

    # if target function
    if 'f' in function_dict[key]:
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

    if params['size']:
        gp = GeneticProgramming(rng=rng,
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
params['AFSPO'] = args.AFSPO
params['size'] = args.size
params['IA'] = args.use_interval_arithmetic

last_folder = cf.get_folder(params) + '/'

params['count_asymptotes'] = False
params['save_pop_data'] = False

print('last_folder', last_folder)

if len(run_list) <= run_index:

    print('WARNING: run_index out of range. If redos, this is expected.')
    exit()

start = time.time()

run_single(rng=np.random.RandomState(run_list[run_index] + exp), pop_size=population_size,
           primitive_set=primitive_set,
           terminal_set=terminal_set, prob_mutate=1.,
           prob_xover=0.,
           mutation_param=mutation_param, rep=run_list[run_index],
           output_path=os.path.join(os.environ['GP_DATA'], 'AFPO/experiments/' + str(exp) + '/' + key + '/'+last_folder),
           output_file='fitness_data_rep' + str(run_list[run_index]) + '.csv', **params)

print(time.time() - start)
