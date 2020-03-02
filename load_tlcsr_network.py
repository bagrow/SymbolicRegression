import numpy as np
import pandas as pd

import os

def load_tlcsr_network(exp, rep, test_index, network):

    filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'best_ind_model_weights_rep'+str(rep)+'_test_index'+str(test_index)+'.h5')

    network.load_weights(filename)

    return network


if __name__ == '__main__':

    from TlcsrNetwork import TlcsrNetwork
    from GeneticProgramming.Tree import Tree

    options = {'use_k-expressions': True,
               'head_length': 15}

    primitive_set = ['*', '+', '-']
    terminal_set = ['x0']

    # if args.use_constants:
    #     terminal_set.append('#f')
    #     terminal_set.append('const_value')

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

    timelimit = 100

    model = TlcsrNetwork(rng=np.random.RandomState(0), #100*args.exp+args.rep),
                         num_data_encoder_inputs=2,
                         primitive_set=primitive_set,
                         terminal_set=terminal_set,
                         # max_decoder_seq_length=10,
                         use_constants=False,#args.use_constants,
                         timelimit=timelimit,
                         options=options)

    exp = 25
    table = [['exp', 'rep', 'test_index', 'dataset', 't', 'RMSE', 'f_hat_t', 'f_hat_t+1']]

    for rep in range(30):

        for test_index in range(6):

            for f in function_order:

                model.network = load_tlcsr_network(exp=exp,
                                                   rep=rep,
                                                   test_index=test_index, 
                                                   network=model.network)

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

                input_function_str = '(- (x0) (x0))'
                for t, (RMSE, f_str) in enumerate(zip(output['errors'], output['equation_str'])):
                    tree = Tree(tree=f_str)
                    output_function_str = tree.get_lisp_string()
                    row = [exp, rep, test_index, f, t, RMSE, input_function_str, output_function_str]
                    input_function_str = output_function_str
                    table.append(row)

            print('.', flush=True, end='')

    filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'figures', 'data_equations_produced.csv')

    df = pd.DataFrame(table[1:], columns=table[0])
    df.to_csv(filename, index=False)