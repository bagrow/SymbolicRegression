# Overwrite (or create) the constants and protected functions
import pickling_setup.protected_functions_writer_fb
import pickling_setup.consts_writer_fb

# Read in the constants and protected functions
from GeneticProgramming.protected_functions import *
from GeneticProgramming.consts import *

# Now import like normal
import GeneticProgramming as GP
from GeneticProgramming.GeneticProgrammingAfpo import GeneticProgrammingAfpo
from GeneticProgramming.GeneticProgrammingRestricted import GeneticProgrammingRestricted
from GeneticProgramming.IndividualRestricted import IndividualRestricted
import GeneticProgramming.data_setup as ds
import GeneticProgramming.common_functions as cf

from get_computation import get_computation_time
import FunctionBuilder as FB

import cma
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

parser.add_argument('-re', '--redos', help='Specific runs to do',
                    type=str, action='store', default='')

# GP settings
parser.add_argument('-r', '--restriction', help='If False, run standard GP instead of restricted.',
                    action='store_false')

# Function Builder Settings
parser.add_argument('-f', '--function_builder', help='If True, run function builder.',
                    action='store_true')
parser.add_argument('-d', '--depth', help='The depth of the tree to be label',
                    action='store', type=int, default=3)
parser.add_argument('-m', '--multiple_networks', help='If False, uses the same network to label each node.',
                    action='store_true')
parser.add_argument('--hidden', help='The number of hidden nodes in each layer as a tuple.',
                    type=int, nargs='+', action='store')
parser.add_argument('--cmaes', help='Use cma-es', action='store_true')
parser.add_argument('--nes', help='Use natrual es', action='store_true')

# General Settings
parser.add_argument('-t', '--timeout', help='Number of seconds after which to stop.',
                    action='store', type=float, default=float('inf'))
parser.add_argument('-fe', '--function_evals', help='Maximum number of function evaluations.',
                    type=float, action='store', default=float('inf'))
parser.add_argument('-npf', '--num_partial_fills', help='The number of partial fills to use in each evaluation',
                    action='store', type=int, default=6)

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

# initialize and run gp
primitive_set = ['+', '*', '-', '%', 'sin2', 'id2']
terminal_set = ['#x', '#f']


def run_single(rng, pop_size, primitive_set, terminal_set,
               prob_mutate, prob_xover, mutation_param,
               rep, output_path, output_file, **params):

    if args.restriction:
        gp = GeneticProgrammingRestricted(rng=rng,
                                          pop_size=pop_size,
                                          primitive_set=primitive_set,
                                          terminal_set=terminal_set,
                                          # this is not data, which is passed
                                          data=dataset,
                                          test_data=dataset_test,
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
                                    test_data=dataset_test,
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
params['FB'] = args.function_builder
params['CMAES'] = args.cmaes
params['NES'] = args.nes
params['D'] = args.depth
params['M'] = args.multiple_networks
params['H'] = args.hidden[0] if args.hidden is not None else 0
params['R'] = args.restriction
params['T'] = args.timeout
params['NPF'] = args.num_partial_fills

last_folder = cf.get_folder(params) + '/'
print('last_folder', last_folder)

params['IA'] = False
params['AFSPO'] = False
params['size'] = False
params['count_asymptotes'] = False
params['save_pop_data'] = False

# Adjust time
if args.timeout == float('inf'):
    timeout = args.timeout

else:
    params['T'] = timeout = get_computation_time(params['T'])

rng = np.random.RandomState(exp)

if args.func in ('test', 'linear', 'quadratic-1', 'quadratic-2', 'quadratic-3'):

    num_data_points = 200*100

    if args.func == 'test' or args.func == 'linear':
        target = lambda x: 2*x[0]

    elif args.func == 'quadratic-1':
        target = lambda x: x[0]**2

    elif args.func == 'quadratic-2':
        target = lambda x: x[0]**2 - 1.

    elif args.func == 'quadratic-3':
        target = lambda x: x[0]**2+2*x[0]+2

    X1 = np.linspace(-1, 1, num_data_points)
    X2 = np.linspace(-20, 3, num_data_points)
    rng.shuffle(X2)
    X = np.vstack((X1, X2))

    dataset = np.vstack((target(X), X1, X2)).T

    # indices_train = rng.choice(num_data_points, size=int(num_data_points*frac_train), replace=False)
    # indices_test = np.array([i for i in range(num_data_points) if i not in indices_train])
    # dataset_train = dataset[indices_train]
    # test_data = dataset[indices_test]

    # dataset = np.array([dataset_train, dataset_train])

    num_vars = 2

    k = 100
    folds = ds.get_k_folds(np.random.RandomState(0), k, dataset)

    dataset_train, dataset_test = ds.get_datasets_from_folds(args.rep, folds)

    val_frac = 0.2

    indices_val = rng.choice(len(dataset_train),
                             size=int(val_frac*len(dataset_train)),
                             replace=False)
    indices_train = [i for i in range(len(dataset_train)) if i not in indices_val]

    dataset_val = np.array([dataset_train[i] for i in indices_val])
    dataset_train = np.array([dataset_train[i] for i in indices_train])

    dataset = [dataset_train, dataset_val]

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

if len(run_list) <= run_index:

    print('WARNING: run_index out of range. If redos, this is expected.')
    exit()

start = time.time()

base_path = os.path.join(os.environ['GP_DATA'], 'function_builder', 'experiments', str(exp), key, last_folder)

os.makedirs(base_path, exist_ok=True)

print('base_path', base_path)

terminal_set_expanded = ['x'+str(i) for i in range(num_vars)] + ['#f']

if args.function_builder:

    assert args.cmaes or args.nes, 'Either cmaes or nes must be True.'

    terminal_set = terminal_set_expanded

    FB.run_function_builder(primitives=primitive_set, terminals=terminal_set, depth=args.depth,
                            dataset=dataset, dataset_test=dataset_test,
                            rep=args.rep, multiple_networks=args.multiple_networks,
                            cmaes=args.cmaes, nes=args.nes, hidden=args.hidden,
                            num_partial_fills=args.num_partial_fills, base_path=base_path,
                            timeout=timeout, function_evals=args.function_evals)
else:

    if args.restriction:

        fake_individual = GP.Individual(rng=rng, primitive_set=primitive_set,
                                        terminal_set=terminal_set, method='full',
                                        depth=args.depth)

        node_dict = fake_individual.get_node_dict()
        locations = list(node_dict.keys())
        locations.remove(())

        restrictions = FB.get_partial_fills(rng, primitive_set, terminal_set_expanded,
                                            locations, args.num_partial_fills)
        params['T'] /= len(restrictions)

        for i, res in enumerate(restrictions):

            params['restrictions'] = res

            run_single(rng=np.random.RandomState(run_list[run_index] + exp), pop_size=population_size,
                       primitive_set=primitive_set,
                       terminal_set=terminal_set, prob_mutate=1.,
                       prob_xover=0.,
                       mutation_param=mutation_param, rep=run_list[run_index],
                       output_path=os.path.join(os.environ['GP_DATA'], base_path, 'restriction'+str(i)),
                       output_file='fitness_data_rep' + str(run_list[run_index]) + '.csv', **params)

    else:

        run_single(rng=np.random.RandomState(run_list[run_index] + exp), pop_size=population_size,
                   primitive_set=primitive_set,
                   terminal_set=terminal_set, prob_mutate=1.,
                   prob_xover=0.,
                   mutation_param=mutation_param, rep=run_list[run_index],
                   output_path=os.path.join(os.environ['GP_DATA'], base_path),
                   output_file='fitness_data_rep' + str(run_list[run_index]) + '.csv', **params)

print(time.time() - start)
