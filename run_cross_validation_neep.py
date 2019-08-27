import GeneticProgramming.data_setup as ds
import NeuroEncodedExpressionProgramming as NEEP
from get_computation import get_computation_time

import numpy as np

import itertools
import argparse

# --------------------------------------------------------- #
#                      PARAMETERS
# --------------------------------------------------------- #

parser = argparse.ArgumentParser()

# this will act as an index for rep_list (offset by 1 though)
# parser.add_argument('rep', help='Number of runs already performed', type=int)

parser.add_argument('index_index', help='The choice of train, val, test', type=int)

parser.add_argument('func', help='Specify the target function', type=str)
# parser.add_argument('exp', help='exp is the experiment number', type=int)


args = parser.parse_args()

rng = np.random.RandomState(0)

if args.func in ('test', 'linear', 'quadratic-1', 'quadratic-2', 'quadratic-3', 'quadratic-4', 'constant-1'):

    num_data_points = 200*100

    if args.func == 'test' or args.func == 'linear':
        target = lambda x: 2*x[0]

    elif args.func == 'quadratic-1':
        target = lambda x: x[0]**2

    elif args.func == 'quadratic-2':
        target = lambda x: x[0]**2 - 1.

    elif args.func == 'quadratic-3':
        target = lambda x: x[0]**2+2*x[0]+2

    elif args.func == 'quadratic-4':
        target = lambda x: 0.5*x[0]**2 + np.sqrt(2)*x[0] - 1.

    elif args.func == 'constant-1':
        target = lambda x: 1.23456789 +0*x[0]

    X1 = np.linspace(-1, 1, num_data_points)
    X2 = np.linspace(-20, 3, num_data_points)
    rng.shuffle(X2)
    X = np.vstack((X1, X2))

    dataset = np.vstack((target(X), X1, X2)).T

sigmas = np.linspace(0.01, 4, 25)

k = 5
folds = ds.get_k_folds(np.random.RandomState(0), k, dataset)

indices = [(i, j, k) for i, val in enumerate(folds) for j, test in enumerate(folds) for k, s in enumerate(sigmas) if i != j]

i, j, k = indices[args.index_index]
sigma = sigmas[k]
print(i, j, sigma)

val = folds[i]
test = folds[j]

train = np.array(list(itertools.chain.from_iterable([x for k, x in enumerate(folds) if k not in (i, j)])))

dataset = [train, val]

timeout = get_computation_time(18000)

test_error = NEEP.run_neuro_encoded_expression_programming(rep=0, num_hidden=40, head_size=30, primitives=['*', '+', '-', '%', 'sin'],
                                                           terminals=['x0', 'x1', '#f'], dataset=dataset, dataset_test=test,
                                                           timeout=timeout, function_evals=float('inf'), base_path=None,
                                                           num_time_steps=10, sigma=sigma)

print('test error', test_error)
