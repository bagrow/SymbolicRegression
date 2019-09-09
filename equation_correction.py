import GeneticProgramming as GP
from GeneticProgramming.protected_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma

import argparse
import os

def get_value(u, v, w, input, hidden, activation):
    """Compute the output given the input, hidden node
    values, and the weights.

    Parameters
    ----------
    u : 2D np.array
        The weights between input layer and hidden layer.
    v : 2D np.array
        The weights between hidden layer and itself.
    w : 2D np.array
        The weights between the hidden layer and output layer.
    input : 1D np.array
        The input value for the input nodes.
    hidden : 1D np.array
        The current values of the hidden nodes.
    activation : function
        The activation function.

    Returns
    -------
    output : 1D np.array
        The value of the nodes in the output layer.
    hidden: 1D np.array
        The new values of the nodes in the hidden layer.
    """

    hidden = activation(np.matmul(input, u) + np.matmul(hidden, v))
    output = activation(np.matmul(hidden, w))

    return output, hidden


def evalutate_corrector_neural_network(w, signed_error, hidden_values, hidden_weights, activation):
    """Evaluate neural network that decides how to change the function.

    Parameters
    ----------
    w : 1D np.array
        The weights of the neural network as a one dimensional
        array.
    signed_error : float
        The signed error of the current tree.
    hidden_values : 1D np.array
        The values of the hidden nodes.
    activation : function
        The activation function.

    Returns
    -------
    index : int
        The argmax of the output layer. This specifies the
        way to alter the function.
    new_hidden_value : np.array
        The hidden values after evalution.
    """

    num_hidden = len(hidden_values)

    # Take one node for the signed error.
    u = w[:num_hidden].reshape(1, num_hidden)

    # Get recurrent weights
    v = hidden_weights

    # Get the output weights. Output is one of three nodes.
    w = w[num_hidden:].reshape((num_hidden, 3))

    output, new_hidden_value = get_value(u, v, w, [signed_error], hidden_values, activation)

    index = np.argmax(output)

    return index, new_hidden_value


def update_equation(function_string, dataset, w, signed_error, hidden_values, hidden_weights,
                    activation, adjustment, constant):
    """

    Parameters
    ----------
    w : 1D np.array
        The weights of the neural network as a one dimensional
        array.
    signed_error : float
        The signed error of the current tree.
    hidden_values : 1D np.array
        The values of the hidden nodes.
    activation : function
        The activation function.
    adjustment : float
        The amount of change that is possible to do
        to constant. In other words, the returned
        new_constant could be constant, constant+adjustment
        or constant-adjustment.
    constant : float
        The value to adjust a specific constant by.

    Returns
    -------
    error : float
        The RMS error of the adjusted function
    hidden_values : np.array
        The hidden values of the neural network
        as returned by evalutate_corrector_neural_network.
    new_constant : float
        The updated constant.
    """


    index, hidden_values = evalutate_corrector_neural_network(w, signed_error, hidden_values, hidden_weights,
                                                              activation)

    if index == 0:
        new_constant = constant + adjustment
        # print('+')

    elif index == 1:
        new_constant = constant - adjustment
        # print('-')

    elif index == 2:
        new_constant = constant
        # print('0')

    else:
        print('Unspecified option. index =', index)
        exit()

    f = eval('lambda x: '+function_string+'+'+str(new_constant))

    x = dataset[:, 1:]
    y = dataset[:, 0]

    error = np.sqrt(np.mean(np.power(f(x.T)-y, 2)))
    signed_error = np.mean(y-f(x.T))

    return error, signed_error, hidden_values, new_constant


def run_equation_corrector(function_string, dataset, w, hidden_values, hidden_weights, activation,
                           adjustment, constant=0, num_iterations=5):

    # init signed error
    x = dataset[:, 1:]
    y = dataset[:, 0]
    f = eval('lambda x: '+function_string+'+'+str(constant))
    signed_error = np.mean(y-f(x.T))

    for _ in range(num_iterations):

        error, signed_error, hidden_values, constant = update_equation(function_string, dataset, w,
                                                                       signed_error, hidden_values,
                                                                       hidden_weights, activation,
                                                                       adjustment, constant)
        # print('error', error, 'signed_error', signed_error, 'constant', constant)

    return error, constant


def cma_es_function(w, rng, depth, hidden_values, hidden_weights, activation, adjustment,
                    num_iterations, datasets, return_all_errors=False):
    """Function that is passed to CMA-ES."""

    errors = []

    for function_string in datasets:

        constant = 0
        
        dataset = datasets[function_string]

        error, constant = run_equation_corrector(function_string, dataset, w, hidden_values, hidden_weights,
                                                 activation, adjustment, constant, num_iterations)

        errors.append(error)
    
    if return_all_errors:
        return errors

    else:
        return np.mean(errors)


def train_equation_corrector(rng, save_loc):

    hidden_values = rng.uniform(-1, 1, size=10)
    hidden_weights = rng.uniform(-1, 1, size=(len(hidden_values), len(hidden_values)))
    activation = np.tanh

    weights = rng.uniform(-1, 1, size=len(hidden_values)+len(hidden_values)*3)
    adjustment = 1
    
    # get data
    num_targets = 50
    num_base_function_per_target = 2
    depth = 3

    sigma = 1.
    function_evals = float('inf')
    seed = args.rep + 1
    # timeout = get_computation_time(18000)
    timeout = 360
    num_iterations = 50

    datasets = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth)
    datasets_test = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth)

    xopt, es = cma.fmin2(cma_es_function, weights, sigma,
                     args=(rng, depth, hidden_values, hidden_weights, activation, adjustment,
                           num_iterations, datasets),
                     options={'maxfevals': function_evals,
                              # 'ftarget': 1e-10,
                              'tolfun': 0,
                              # 'tolfunhist': 0,
                              'seed': seed,
                              'verb_log': 0,
                              'timeout': timeout},
                     restarts=0)

    test_errors = cma_es_function(xopt, rng, depth, hidden_values, hidden_weights, activation, adjustment,
                                 num_iterations, datasets, return_all_errors=True)

    # save the best individual
    # ['test error', 'trained weights', 'untrained weights', 'initial hidden values']
    data = [[test_errors], xopt, hidden_weights.flatten(), hidden_values]
    df = pd.DataFrame(data).transpose()
    df.to_csv(save_loc, header=['test error', 'trained weights', 'untrained weights', 'initial hidden values'])

    plt.hist(test_errors)
    plt.show()


def get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth):
    """Generate a dataset for multiple functions. Keep track of
    the base function(s) connected with the dataset.

    Parameters
    ----------
    rng : random number generator
        Example: np.random.RandomState(0)
    num_targets : int
        The number of target functions.
    num_base_function_per_target : int
        The number of base functions to make for
        each target funtion.
    depth : int
        The max depth of the trees (that represent
        the target functions).

    Returns
    -------
    datasets : dict
        Dictionary with key=base function and
        value=dataset.
    """

    primitives = ['*', '+', '%', '-']
    terminals = ['#x', '#f']

    # Pick the number of input variables
    # between 1 and 6.
    num_vars = rng.choice(5)+1

    targets = [GP.Individual(rng=rng, primitive_set=primitives, terminal_set=terminals, num_vars=num_vars,
                             depth=depth, method='grow') for _ in range(num_targets)]

    datasets = {}

    for t in targets:

        offset = rng.choice(list(range(-40, 40)), size=num_base_function_per_target)

        base_file = os.path.join(os.environ['GP_DATA'], 'tree')
        
        for o in offset:

            base_function_string = t.convert_lisp_to_standard_for_function_creation()
            function_string = base_function_string+'+'+str(o)

            function = eval('lambda x: '+function_string)

            # Make inputs
            x = np.array([rng.uniform(-1, 1, size=100) for _ in range(num_vars)]).T
            # print('x', x)
            # print('function(x.T)', np.array([function(x.T)]).T)
            dataset = np.hstack((np.array([function(x.T)]).T, x))
            # print(dataset)

            datasets[base_function_string] = dataset

    return datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # this will act as an index for rep_list (offset by 1 though)
    parser.add_argument('rep', help='Number of runs already performed', type=int)

    args = parser.parse_args()

    rng = np.random.RandomState(args.rep)

    save_loc = os.path.join(os.environ['GP_DATA'], 'function_builder', 'equation_adjuster', 'best_ind_rep'+str(args.rep)+'.csv')

    train_equation_corrector(rng, save_loc)

    # signed_error = 1
    # hidden_values = rng.uniform(-1, 1, size=10)
    # activation = np.tanh

    # target = lambda x: 2+x
    # x = np.linspace(-1, 1, 100)
    # y = target(x)
    # dataset = np.vstack((y, x)).T

    # w = rng.uniform(-1, 1, size=1+len(hidden_values)**2+len(hidden_values)*3)
    # adjustment = 1
    
    # # get data
    # num_targets = 5
    # num_base_function_per_target = 2
    # depth = 3

    # get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth)

