from load_tlcsr_network import load_tlcsr_network

from TlcsrNetwork import TlcsrNetwork
from GeneticProgramming.Tree import Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

np.random.seed(0)

exp = 25
# rep = 1
# test_index = 5

function_order = ['Nguyen-4',
                  'Koza-1',
                  'Koza-2',
                  'Koza-3',
                  'Nguyen-1',
                  'Nguyen-3',
                  'Validation']

function_strs = {'Nguyen-4': 'x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Nguyen-4
                 'Koza-1': 'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
                 'Koza-2': 'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
                 'Koza-3': 'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
                 'Nguyen-1': 'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
                 'Nguyen-3': 'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]', # Nguyen-3
                 'Validation': 'x[0]**4 + x[0]'
                }

options = {'use_k-expressions': True,
           'head_length': 15,
           # 'eq_encoder_only': True}
           'data_encoder_only': True}

primitive_set = ['*', '+', '-']
terminal_set = ['x0']

model = TlcsrNetwork(rng=np.random.RandomState(0), #100*args.exp+args.rep),
                     num_data_encoder_inputs=2,
                     primitive_set=primitive_set,
                     terminal_set=terminal_set,
                     # max_decoder_seq_length=10,
                     use_constants=False,
                     timelimit=100,
                     options=options)

table = [['exp', 'rep', 'test_index', 'dataset', 'Shuffled input?', 'network_type', 'error_best', 'equation_best'] + ['hidden_state['+str(i)+']' for i in range(16)]]

for rep in range(30):

    for test_index in range(6):

        for j, f in enumerate(function_order):
            f = function_order[test_index]

            if j > 0:
                break

            for network_type in ['random', 'trained']:

                if network_type == 'trained':
                    model.network = load_tlcsr_network(exp=exp,
                                                       rep=rep,
                                                       test_index=test_index, 
                                                       network=model.network)

                else:
                    model.set_weights(np.random.uniform(-1, 1, size=1478))

                # get dataset
                x = np.linspace(-1, 1, 20)[None, :]
                target = eval('lambda x: '+function_strs[f])
                y = target(x)

                # initialize f_hat to the zero function
                initial_f_hat = lambda x: 0*x[0]
                initial_f_hat_seq = ['-', 'x0', 'x0']

                output = model.evaluate(x, y, initial_f_hat, initial_f_hat_seq,
                                        return_equation=False,
                                        return_equation_str=True,
                                        return_decoded_list=False,
                                        return_errors=True)

                hidden_state = np.hstack((output['state_h1'], output['state_h2'])).flatten()

                row = [exp, rep, test_index, f, False, network_type, output['error_best'], output['equation_best']] + list(hidden_state)

                table.append(row)

                for _ in range(10):

                    # D = np.array([(xi,yi) for xi, yi in zip(x[0], y)])
                    # np.random.shuffle(D)

                    # x_shuffled = D[:, 0][None,:]
                    # y_shuffled = D[:, 1]

                    # assert np.all(target(x_shuffled)==y_shuffled), 'Shuffling failure!'

                    # x = np.random.uniform(-1, 1, 20)[None, :]
                    np.random.shuffle(x[0])
                    y = target(x)

                    assert np.any(np.linspace(-1, 1, 20)[None, :]!=x), 'Not shuffling'


                    output = model.evaluate(x, y,
                                            initial_f_hat, initial_f_hat_seq,
                                            return_equation=False,
                                            return_equation_str=True,
                                            return_decoded_list=False,
                                            return_errors=True)

                    hidden_state = np.hstack((output['state_h1'], output['state_h2'])).flatten()

                    row = [exp, rep, test_index, f, True, network_type, output['error_best'], output['equation_best']] + list(hidden_state)

                    table.append(row)

                    print('.', flush=True, end='')

filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'figures', 'data_encoder_outputs.csv')

df = pd.DataFrame(table[1:], columns=table[0])
df.to_csv(filename, index=False)