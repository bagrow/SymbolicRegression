"""Train seq2seq model (input=dataset and previous equation, output=equation) to do
symbolic regression.
"""

from seq2seq_second import seq2seq
from CmaesTrainsNn import CmaesTrainsNn
import GeneticProgramming as GP

import numpy as np

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('rep', help='Number of runs already performed', type=int)
parser.add_argument('exp', help='Experiment number. Used in save location', type=int)
parser.add_argument('--use_kexpressions', action='store_true')
parser.add_argument('--genetic_programming', action='store_true')

args = parser.parse_args()

if args.use_kexpressions:
    options = {'use_k-expressions': True,
               'head_length': 15}

else:
    options = None

primitive_set = ['*', '+', '-']
terminal_set = ['x0']

s2s = seq2seq(num_data_encoder_tokens=2,
              primitive_set=primitive_set,
              terminal_set=terminal_set,
              max_decoder_seq_length=10,
              timelimit=10,
              options=options)

x = np.linspace(-1, 1, 20)[None, :]
f = lambda x: x[0]**4 + x[0]**3 + x[0]**2 + x[0]
y = f(x)

f_val = lambda x: x[0]**4 + x[0]**3 + x[0]**2 + x[0]
y_val = f(x)

f_test = lambda x: x[0]
y_test = f_test(x)

rep = args.rep
exp = args.exp

max_compute = 10**10

if args.genetic_programming:

    # format datasets
    train_dataset = np.array([y, x[0]]).T
    val_dataset = np.array([y_val, x[0]]).T
    test_dataset = np.array([y_test, x[0]]).T

    # get the name from list rather than benchmark_datasets,
    # which is a dict
    rng = np.random.RandomState(args.rep+100*args.exp)

    output_path = os.path.join(os.environ['EE_DATA'], 'experiment'+str(args.exp), 'gp')

    # the population from the previous run of
    # genetic programming
    prev_pop = None

    # dataset is the training dataset and validation dataset
    dataset = [train_dataset, val_dataset]
    test_data = test_dataset

    output_file = 'fitness_data_rep' + str(args.rep) + '.csv'

    num_vars = 1

    params = {'max_compute': max_compute}

    gp = GP.GeneticProgrammingAfpo(rng=rng,
                                   pop_size=100,
                                   max_gens=60000,
                                   primitive_set=primitive_set,
                                   terminal_set=terminal_set,
                                   # this is not data, which is passed
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

else:

    fitter = CmaesTrainsNn(rep=rep, exp=exp,
                           model=s2s,
                           x=x, y=y,
                           x_test=x, y_test=y_test, test_dataset_name='test',
                           timelimit=10,
                           options=options)

    cmaes_options = {'popsize': 100,
                     'tolfun': 0}  # toleration in function value

    fitter.fit(max_FLoPs=max_compute, sigma=0.5, cmaes_options=cmaes_options)