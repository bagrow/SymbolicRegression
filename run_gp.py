from GeneticProgrammingAfpo import GeneticProgrammingAfpo
import GeneticProgrammingAfpo.data_setup as ds
from GeneticProgrammingAfpo.consts import *
from GeneticProgrammingAfpo.protected_functions import *

import numpy as np

import os
import argparse
import time
import sys

sys.setcheckinterval(1000000)     # If running thread, this is not a good idea

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

num_vars = 1 if key not in number_of_input_variables else number_of_input_variables[key]
print('num_vars', num_vars)

noise_std = 0.1

# --------------------------------------------------------- #
#                      END PARAMETERS
# --------------------------------------------------------- #

# always use the same seed for each run in exp
test_data = ds.get_datasets(rng=np.random.RandomState(exp),
                            f=function_dict[key]['f'],
                            A=function_dict[key]['a'],
                            B=function_dict[key]['b'],
                            noise_percent=None,
                            noise_std=noise_std,
                            data_size=int(100000 / num_vars))[0]

# initialize and run gp
primitive_set = ['+', '*', '-', '%']
terminal_set = ['#x', '#f']


def run_single(rng, pop_size, primitive_set, terminal_set, test_data,
               prob_mutate, prob_xover, num_vars, max_depth, mutation_param,
               rep, output_path, output_file, **params):

    dataset = ds.get_datasets(rng=np.random.RandomState(rep),
                              f=function_dict[key]['f'],
                              A=function_dict[key]['a'],
                              B=function_dict[key]['b'],
                              noise_percent=None,
                              noise_std=noise_std,
                              data_size=function_dict[key]['size'])

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
                                max_depth=max_depth,
                                mutation_param=mutation_param,
                                # parameters below
                                **params)

    info = gp.run(rep=rep,
                  output_path=output_path,
                  output_file=output_file)

    return info


params = {'save_pop_data': True if run_list[run_index] < 10 else False}

if len(run_list) <= run_index:

    print('WARNING: run_index out of range. If redos, this is expected.')
    exit()

start = time.time()

run_single(rng=np.random.RandomState(run_list[run_index] + exp), pop_size=population_size,
           primitive_set=primitive_set,
           terminal_set=terminal_set, test_data=test_data, prob_mutate=1.,
           prob_xover=0., num_vars=num_vars, max_depth=max_depth,
           mutation_param=mutation_param, rep=run_list[run_index],
           output_path=os.path.join(os.environ['GP_DATA'], 'AFPO/experiments/' + str(exp) + '/' + key + '/'),
           output_file='fitness_data_rep' + str(run_list[run_index]) + '.csv', **params)

print(time.time() - start)
