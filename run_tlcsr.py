"""Train seq2seq model (input=dataset and previous equation, output=equation) to do
symbolic regression.
"""

from TlcsrNetwork import TlcsrNetwork
from Tlcsr import Tlcsr
import GeneticProgramming as GP
from GeneticProgramming.IndividualManyTargetData import IndividualManyTargetData

import numpy as np

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('rep', help='Number of runs already performed', type=int)
parser.add_argument('exp', help='Experiment number. Used in save location', type=int)
parser.add_argument('--use_kexpressions', action='store_true', help='Use k-expressions to interpret input and output to TLC-SR')
parser.add_argument('--genetic_programming', action='store_true', help='Do genetic programming')
parser.add_argument('--single_target', action='store_true', help='Use only one target function to create train, validation, test datasets')
parser.add_argument('--simultaneous_targets', action='store_true', help='Use multiple target functions for train, validation, test datasets')
parser.add_argument('--use_constants', action='store_true', help='Included constants with two nodes: one for one-hot and other for constant value')
parser.add_argument('--inconsistent_x', action='store_true', help='Random (uniformly dist) selection of x-values')
# parser.add_argument('--shuffle_x', action='store_true', help='Shuffle x-values before input into NN')
parser.add_argument('--use_benchmarks', action='store_true', help='Use the benchmark functions for datasets')
parser.add_argument('--test_index', type=int, help='Index of the target function to use to create test dataset. Others are used for training if --simultaneous_targets')

args = parser.parse_args()
print(args)

assert args.test_index is not None, 'Must use --test_index'

assert (args.simultaneous_targets and args.single_target) == False, 'Cannot use --simultaneous_targets and --single_target at the same time'

if args.use_kexpressions:
    options = {'use_k-expressions': True,
               'head_length': 15}   # this defines length eq output from NN

else:
    options = None

primitive_set = ['*', '+', '-']
terminal_set = ['x0']

if args.use_constants:
    terminal_set.append('#f')
    terminal_set.append('const_value')

timelimit = 100

if args.use_benchmarks:
    function_strs = ['x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Nguyen-4
                     'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
                     'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
                     'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
                     'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
                     'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]' # Nguyen-3
                    ]

    train_function_strs = [f for i, f in enumerate(function_strs) if i != args.test_index]

else:
    train_function_strs = ['x[0]',
                           '2*x[0]',
                           'x[0]**2',
                           'x[0]**2 + x[0]',
                           'x[0]**3']


functions = [eval('lambda x: '+f_str) for f_str in train_function_strs]

if args.inconsistent_x:
    gen_x_values = lambda: np.array(sorted(np.random.uniform(-1, 1, size=20))) 

else:
    gen_x_values = lambda: np.linspace(-1, 1, 20)

if not args.single_target:

    X_train = [gen_x_values()[None, :] for _ in range(len(functions))]
    Y_train = [f(x) for x, f in zip(X_train, functions)]

    x_val = gen_x_values()[None, :]
    f_val = lambda x: x[0]**4 + x[0]
    y_val = f_val(x_val)

    x_test = gen_x_values()[None, :]

if args.use_benchmarks:

    assert 0 <= args.test_index < len(function_strs), '--test_index must be between 0 and '+str(len(function_strs)-1)

    f_test = eval('lambda x: '+function_strs[args.test_index])

else:

    test_function_strs = ['x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]' # Nguyen-4
                          'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
                          'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
                          'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
                          'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
                          'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]', # Nguyen-3
                         ]

    assert 0 <= args.test_index < len(test_function_strs), '--test_index must be between 0 and '+str(len(test_function_strs)-1)

    f_test = eval('lambda x: '+test_function_strs[args.test_index])

if args.single_target:

    X_train = [np.array(sorted(np.random.uniform(-1, 1, size=20)))[None, :]]
    Y_train = [f_test(x) for x in X_train]

    x_val = np.array(sorted(np.random.uniform(-1, 1, size=20)))[None, :]
    y_val = f_test(x_val)

    x_test = np.array(sorted(np.random.uniform(-1, 1, size=20)))[None, :]

y_test = f_test(x_test)

rep = args.rep
exp = args.exp

max_effort = 5*2*10**9

rng = np.random.RandomState(args.rep+100*args.exp)

if args.genetic_programming:

    if args.simultaneous_targets:

        train_dataset = []

        for i, (x_train, y_train) in enumerate(zip(X_train, Y_train)):

            # format datasets
            train_dataset.append(np.array([y_train, x_train[0]]).T)

        val_dataset = np.array([y_val, x_val[0]]).T
        test_dataset = np.array([y_test, x_test[0]]).T

        output_path = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(args.exp), 'gp')

        # dataset is the training dataset and validation dataset
        # This is how the datasets are handle for GP.
        # Perhaps I should update this.
        dataset = [train_dataset, val_dataset]
        test_data = test_dataset

        output_file = 'fitness_data_rep' + str(args.rep) + '_train_all_at_once_test_index'+str(args.test_index)+'.csv'

        num_vars = 1

        # max_effort as set above is per training dataset
        params = {'max_effort': max_effort*len(X_train)}

        gp = GP.GeneticProgrammingAfpo(rng=rng,
                                       pop_size=100,
                                       max_gens=600000, # can't set to inf since range(max_gens) used
                                       primitive_set=primitive_set,
                                       terminal_set=terminal_set,
                                       data=dataset,
                                       test_data=test_data,
                                       prob_mutate=1,
                                       prob_xover=0,
                                       num_vars=num_vars,
                                       mutation_param=2,
                                       individual=IndividualManyTargetData,
                                       # parameters below
                                       **params)

        info = gp.run(rep=args.rep,
                      output_path=output_path,
                      output_file=output_file)

    else:   # for single target

        for i, (x_train, y_train) in enumerate(zip(X_train, Y_train)):

            # format datasets
            train_dataset = np.array([y_train, x_train[0]]).T
            val_dataset = np.array([y_val, x_val[0]]).T
            test_dataset = np.array([y_test, x_test[0]]).T

            output_path = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(args.exp), 'gp')

            # the population from the previous run of
            # genetic programming
            prev_pop = None

            # dataset is the training dataset and validation dataset
            dataset = [train_dataset, val_dataset]
            test_data = test_dataset

            output_file = 'fitness_data_rep' + str(args.rep) + '_train'+str(args.test_index)+'_test_index'+str(args.test_index)+'.csv'

            num_vars = 1

            # max_effort as set above is per training dataset
            params = {'max_effort': max_effort*len(X_train)}

            gp = GP.GeneticProgrammingAfpo(rng=rng,
                                           pop_size=100,
                                           max_gens=600000,
                                           primitive_set=primitive_set,
                                           terminal_set=terminal_set,
                                           data=dataset,
                                           test_data=test_data,
                                           prob_mutate=1,
                                           prob_xover=0,
                                           num_vars=num_vars,
                                           mutation_param=2,
                                           # parameters below
                                           **params)

            if prev_pop is not None:
                gp.pop = prev_pop

            info = gp.run(rep=args.rep,
                          output_path=output_path,
                          output_file=output_file)

            prev_pop = gp.pop

else:   # for TLC-SR

    save_loc = os.path.join(os.environ['EE_DATA'], 'experiment'+str(args.exp))

    model = TlcsrNetwork(rng=np.random.RandomState(100*args.exp+args.rep),
                         num_data_encoder_inputs=2, # (xi, ei) -> data encoder
                         primitive_set=primitive_set,
                         terminal_set=terminal_set,
                         use_constants=args.use_constants,
                         timelimit=timelimit,
                         options=options)

    fitter = Tlcsr(rep=rep, exp=exp,
                   model=model,
                   X_train=X_train, Y_train=Y_train,
                   x_val=x_val, y_val=y_val,
                   x_test=x_test, y_test=y_test,
                   test_dataset_name='test_index'+str(args.test_index),
                   timelimit=timelimit,
                   simultaneous_targets=args.simultaneous_targets,
                   options=options)

    cmaes_options = {'popsize': 100,
                     'tolfun': 0}  # tolerance in function value

    fitter.fit(max_effort=max_effort*len(Y_train),
               sigma=0.5,
               cmaes_options=cmaes_options,
               save_loc=save_loc)
