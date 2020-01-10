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

timelimit = 100

s2s = seq2seq(num_data_encoder_tokens=2,
              primitive_set=primitive_set,
              terminal_set=terminal_set,
              # max_decoder_seq_length=10,
              timelimit=timelimit,
              options=options)

function_strs = ['x[0]', '2*x[0]', 'x[0]**2', 'x[0]**2 + x[0]', 'x[0]**3']
functions = [eval('lambda x: '+f_str) for f_str in function_strs]

x_train = np.linspace(-1, 1, 20)[None, :] 
Y_train = [f(x_train) for f in functions]

x_val = np.linspace(-1, 1, 20)[None, :]
f_val = lambda x: x[0]**4
y_val = f_val(x_val)

x_test = np.linspace(-1, 1, 20)[None, :] 
f_test = lambda x: x[0]**4 + x[0]**3 + x[0]**2 + x[0]
y_test = f_test(x_test)

rep = args.rep
exp = args.exp

max_compute = 2*10**9

rng = np.random.RandomState(args.rep+100*args.exp)

if args.genetic_programming:

    for i, y_train in enumerate(Y_train):

        # format datasets
        train_dataset = np.array([y_train, x_train[0]]).T
        val_dataset = np.array([y_val, x_val[0]]).T
        test_dataset = np.array([y_test, x_test[0]]).T

        output_path = os.path.join(os.environ['EE_DATA'], 'experiment'+str(args.exp), 'gp')

        # the population from the previous run of
        # genetic programming
        prev_pop = None

        # dataset is the training dataset and validation dataset
        dataset = [train_dataset, val_dataset]
        test_data = test_dataset

        output_file = 'fitness_data_rep' + str(args.rep) + '_train'+str(i)+'.csv'

        num_vars = 1

        params = {'max_compute': max_compute}

        gp = GP.GeneticProgrammingAfpo(rng=rng,
                                       pop_size=100,
                                       max_gens=60000,
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

else:

    fitter = CmaesTrainsNn(rep=rep, exp=exp,
                           model=s2s,
                           x_train=x_train, Y_train=Y_train,
                           x_val=x_val, y_val=y_val,
                           x_test=x_test, y_test=y_test, test_dataset_name='test',
                           timelimit=timelimit,
                           options=options)

    cmaes_options = {'popsize': 100,
                     'tolfun': 0}  # toleration in function value

    fitter.fit(max_FLoPs=max_compute*len(Y_train), sigma=0.5, cmaes_options=cmaes_options)