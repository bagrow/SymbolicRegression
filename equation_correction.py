import GeneticProgramming as GP
from GeneticProgramming.protected_functions import *
from get_computation import get_computation_time, get_time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma

import argparse
import os
import time


def get_variant_dict(variant_string):

    variant_dict = {}

    for x in variant_string.split('_'):

        if x[-1] == '1':
            value = True

        elif x[-1] == '0':
            value = False

        else:
            print('HOW DO I CONVERT THIS!~!!!>!>!>???', x)
            exit()        

        variant_dict[x[:-1]] = value

    # check all variants are there
    for v in ['MO', 'RO', 'PE', 'ED']:
        if v not in variant_dict:
            print('Not all variant modifiers accounted for.', v, 'is missing.')
            exit()

    return variant_dict


class EquationAdjustor:

    def __init__(self, initial_hidden_values, hidden_weights, activation, horizontal,
                 initial_adjustment, initial_parameter, num_adjustments, fixed_adjustments,
                 variant_string):
        """Initialize. Note that runnning this function does not create a full
        EquationAdjuster because it does not have input and output weights.

        Parameters
        ----------
        initial_hidden_values : 1D np.array
            The values of the hidden nodes.
        hidden_weights : 1D np.array
            The hidden weights, which do not change.
        activation : function
            The activation function.
        horizontal : bool
            If True, the nn will be used to change
            the base function by a horizontal shift.
            Otherwise, a vertical shift.
        initial_adjustment : float
            The initial amount of change that is possible to do
            to self.parameter. In other words, each time
            self.update_equation is called, self.parameter
            may stay same, increase by self.adjustment
            or decrease by self.adjustment.
        initial_parameter : float
            The initial value of the parameter used to offset
            the base function.
        fixed_adjustments : bool
            If true, the size of the adjustments are constant.
        """

        self.variant = get_variant_dict(variant_string)
        print(self.variant)
        
        self.initial_hidden_values = initial_hidden_values
        self.hidden_weights = hidden_weights
        self.activation = activation

        self.initial_adjustment = initial_adjustment
        self.initial_parameter = initial_parameter

        self.fixed_adjustments = fixed_adjustments

        self.horizontal = horizontal

        if self.horizontal:
            self.adjust_function = lambda function_string, c: eval('lambda x: '+function_string.replace('x[0]', 'np.add(x[0],'+str(c)+')'))
            self.num_output = 2 if self.variant['MO'] else 1

        else:   # vertical shift
            self.adjust_function = lambda function_string, c: eval('lambda x: '+function_string+'+'+str(c))
            self.num_output = 1

        self.set_num_adjustments(num_adjustments)


    def set_num_adjustments(self, num_adjustments):

        self.num_adjustments = num_adjustments

        if self.fixed_adjustments:
            self.step = 0.

        else:
            self.step = self.initial_adjustment/self.num_adjustments


    def reinitialize(self, function_string, dataset):
        """For all variables that change during
        equation adjustment, set them to their 
        initial values."""

        self.hidden_values = self.initial_hidden_values
        self.adjustment = self.initial_adjustment
        self.parameter = self.initial_parameter

        # In case index is passed into nn
        self.index = 0

        # init signed error
        x = dataset[:, 1:]
        y = dataset[:, 0]

        f = self.adjust_function(function_string, self.parameter)

        self.errors = [np.sqrt(np.mean(np.power(f(x.T)-y, 2)))]
        self.signed_error = np.mean(y-f(x.T))

        self.cpu_start_time = time.process_time()
        self.cpu_time = []


    def get_value(self, input_weights, hidden_weights, output_weights, input):
        """Compute the output given the input, hidden node
        values, and the weights.

        Parameters
        ----------
        input_weights : 2D np.array
            The weights between input layer and hidden layer.
        hidden_weights : 2D np.array
            The weights between hidden layer and itself.
        output_weights : 2D np.array
            The weights between the hidden layer and output layer.
        input : 1D np.array
            The input value for the input nodes.

        Returns
        -------
        output : 1D np.array
            The value of the nodes in the output layer.
        """

        self.hidden_values = self.activation(np.matmul(input, input_weights))# + np.matmul(self.hidden_values, hidden_weights))
        output = self.activation(np.matmul(self.hidden_values, output_weights))

        return output


    def evaluate_corrector_neural_network(self, w, error, signed_error, prev_error, prev_index, output):
        """Evaluate neural network that decides how to change the function.

        Parameters
        ----------
        w : 1D np.array
            The weights of the neural network as a one dimensional
            array.
        signed_error : float
            The signed error of the current tree.

        Returns
        -------
        index : int
            The argmax of the output layer. This specifies the
            way to alter the function.
        """

        num_hidden = len(self.hidden_values)

        if self.horizontal:
            
            input = [error]

            # prev error
            if self.variant['PE']:
                input.append(prev_error)

            # error difference
            if self.variant['ED']:
                input.append(error - prev_error)

            # if recursive output
            if self.variant['RO']:

                # if multiple output
                if self.variant['MO']:
                    input.extend(np.eye(2)[int(prev_index)])

                else:
                    input.append(output)

        else:
            input = [signed_error]

        # Take one node for the signed error.
        input_weights = w[:num_hidden*len(input)].reshape((len(input), num_hidden))

        # Get recurrent weights
        hidden_weights = self.hidden_weights

        # Get the output weights. Output is one of three nodes.
        output_weights = w[num_hidden*len(input):].reshape((num_hidden, self.num_output))

        new_output = self.get_value(input_weights, hidden_weights, output_weights, input)

        index = np.argmax(new_output)

        return index, new_output


    def update_equation(self, function_string, dataset, w):
        """

        Parameters
        ----------
        w : 1D np.array
            The weights of the neural network as a one dimensional
            array.
        signed_error : float
            The signed error of the current tree.

        Returns
        -------
        error : float
            The RMS error of the adjusted function
        """

        prev_error = self.errors[-2] if len(self.errors) > 1 else self.errors[0]
        output = self.output[0] if hasattr(self, 'output') else 0.

        self.index, self.output = self.evaluate_corrector_neural_network(w, error=self.errors[-1]/self.errors[0],
                                                                         signed_error=self.signed_error/self.errors[0],
                                                                         prev_error=prev_error/self.errors[0], 
                                                                         prev_index=self.index,
                                                                         output=output)

        if self.variant['MO']:

            if self.index == 0:
                self.parameter += self.adjustment

            elif self.index == 1:
                self.parameter -= self.adjustment

            elif self.index == 2:
                pass # keep self.parameter the same

            else:
                print('Unspecified option. index =', index)
                exit()

        else:

            self.parameter += self.output[0]

        f = self.adjust_function(function_string, self.parameter)

        x = dataset[:, 1:]
        y = dataset[:, 0]

        self.errors.append(np.sqrt(np.mean(np.power(y-f(x.T), 2))))
        self.signed_error = np.mean(y-f(x.T))


    def run_equation_corrector(self, function_string, dataset, w):

        for _ in range(self.num_adjustments):

            if self.adjustment < 0:
                print('adjustment is negative! Stopping!')
                exit()

            self.update_equation(function_string, dataset, w)

            self.adjustment -= self.step
            # self.adjustment = abs(self.signed_error)
            self.cpu_time.append(time.process_time()-self.cpu_start_time)

        fitness = self.errors[-1]/self.errors[0]

        return fitness


    def cma_es_function(self, w, rng, datasets, return_all_errors=False):
        """Function that is passed to CMA-ES."""

        training_fitnesses = []
        training_errors = []
        validation_fitnesses = []
        validation_errors = []

        testing = {}

        for function_string, training_dataset, validation_dataset, testing_dataset in datasets['training']:

            # training dataset
            self.reinitialize(function_string, training_dataset)

            training_fitness = self.run_equation_corrector(function_string, training_dataset, w)

            training_fitnesses.append(training_fitness)
            training_errors.append(self.errors[-1])

        # use validation functions to get validation dataset
        for function_string, training_dataset, validation_dataset, testing_dataset in datasets['validation']:

            # validation dataset
            self.reinitialize(function_string, validation_dataset)

            validation_fitness = self.run_equation_corrector(function_string, validation_dataset, w)

            validation_fitnesses.append(validation_fitness)
            validation_errors.append(self.errors[-1])

        if return_all_errors:

            # use validation functions to get validation dataset
            for target_name in datasets['testing']:

                testing[target_name] = {}

                testing_fitnesses = []
                testing_errors = []

                for function_string, training_dataset, validation_dataset, testing_dataset in datasets['testing'][target_name]:

                    # testing dataset
                    self.reinitialize(function_string, testing_dataset)

                    testing_fitness = self.run_equation_corrector(function_string, testing_dataset, w)

                    testing_fitnesses.append(testing_fitness)
                    testing_errors.append(self.errors[-1])

                testing[target_name]['errors'] = self.errors
                testing[target_name]['cpu time'] = self.cpu_time

        global best

        if 'best' in globals():

            if validation_fitness < best[0]:

                best = (np.mean(validation_fitnesses), w)

        else:

            best = validation_fitness

        if return_all_errors:

            errors = {'training': training_errors,
                      'validation': validation_errors,
                      'testing': testing}

            fitnesses = {'training': training_fitnesses,
                         'validation': validation_fitnesses,
                         'testing': testing}

            return errors, fitnesses

        else:

            return np.mean(training_fitnesses)


    def equation_adjuster_from_file(self, filename):
        """Read the data necessary to use eqution
        adjuster from file. File is assume to be
        saved as detailed in train_equation_adjuster().

        Parameters
        ----------
        filename : str
            Location of the file to read from.

        Returns
        -------
        trained_weights : np.array
            The weights for input layer and output
            layer of nn. The array is flat.
        untrained_weights : np.array
            The weights for the hidden layer. This
            array is also 1 dimensional.
        initial_hidden_values : np.array
            The values of the hidden neuron (initially).
        """

        # Remove nans
        rm_nans = lambda x: x[~np.isnan(x)] 

        data = pd.read_csv(filename).iloc[:, 2:].values

        trained_weights, self.initial_hidden_values = rm_nans(data[:, 0]), rm_nans(data[:, 1]), rm_nans(data[:, 2])

        self.hidden_weights = hidden_weights.reshape((len(self.initial_hidden_values, self.initial_hidden_values)))
        self.w = trained_weights

# -------------------------------------------------------------------------- #
#                               End of Class
# -------------------------------------------------------------------------- #


def train_equation_corrector(rep, exp, timeout, fixed_adjustments, horizontal, debug_mode, benchamrk_datasets, num_adjustments, max_shift, variant_string):

    # define parameters
    num_targets = 50
    num_test_targets = 1
    num_base_function_per_target = 1
    depth = 6

    sigma = 2.
    function_evals = float('inf')
    seed = 100*args.exp + args.rep + 1

    activation = np.tanh
    initial_adjustment = 1.
    initial_parameter = 0.

    if not debug_mode:
        timeout, cycles_per_second = get_computation_time(timeout, return_cycles_per_second=True)

    else:
        cycles_per_second = 1.6 * 10**9

    return_all_errors = False

    # get consistent validation set to get best overall EA from experiment
    rng = np.random.RandomState(100*exp)

    datasets_validation_functions_consistent = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth, max_shift, horizontal, fixed_adjustments)

    rng = np.random.RandomState(rep+100*exp)

    hidden_values = rng.uniform(-1, 1, size=10)
    hidden_weights = rng.uniform(-1, 1, size=(len(hidden_values), len(hidden_values)))

    variant = get_variant_dict(variant_string)

    if horizontal:

        if variant['MO']:
            num_output = 2

        else:
            num_output = 1

        num_input = 1

        if variant['RO']:

            if variant['MO']:
                num_input += 2

            else:
                num_input += 1

        if variant['PE']:
            num_input += 1

        if variant['ED']:
            num_input += 1

    else:

        num_input = 1
        num_output = 1

    global best

    # best = (error, weights)
    best = (float('inf'), None)

    weights = rng.uniform(-1, 1, size=num_input*len(hidden_values)+len(hidden_values)*num_output)

    # get data
    datasets = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth, max_shift, horizontal, fixed_adjustments)
    datasets_validation_functions = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth, max_shift, horizontal, fixed_adjustments)
    # datasets_test_functions = get_data_for_equation_corrector(rng, num_test_targets, num_base_function_per_target, depth, max_shift, horizontal, fixed_adjustments)

    all_datasets = {'training': datasets, 'validation': datasets_validation_functions, 'testing': benchamrk_datasets}

    save_loc = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    # Make a file for parameter summary
    summary_data = [('experiment', exp),
                    ('number of hidden nodes', len(hidden_values)),
                    ('number of inputs', num_input),
                    ('number of outputs', num_output),
                    ('number of target functions', num_targets),
                    ('number of target functions for testing', num_test_targets),
                    ('max depth', depth),
                    ('sigma (CMA-ES)', sigma),
                    ('max number of function evaluations', function_evals),
                    ('debug mode', debug_mode),
                    ('horizontal', horizontal),
                    ('max shift', max_shift),
                    ('timeout', timeout),
                    ('fixed adjustment', fixed_adjustments),
                    ('activation', activation.__name__),
                    ('initial adjustment', initial_adjustment),
                    ('initial parameter', initial_parameter),
                    ('number of adjustments', num_adjustments),
                    ('variant string', variant_string)]

    df = pd.DataFrame(summary_data, columns=['Parameters', 'Values'])
    df.to_csv(os.path.join(save_loc, 'summary_exp'+str(exp)+'_'+variant_string+'.csv'))

    # for i, (function_string, training_dataset, validation_dataset, testing_dataset) in enumerate(datasets_test_functions):

    #     table = [[x for x in [label] + list(row)] for label, dataset in zip(['training', 'validation', 'testing'], [training_dataset, validation_dataset, testing_dataset]) for row in dataset]

    #     df = pd.DataFrame(table,
    #                       columns=['dataset', function_string] + ['x['+str(k)+']' for k in range(len(training_dataset[0])-1)])    

    #     df.to_csv(os.path.join(save_loc, 'testing_function_dataset'+str(i)+'_rep'+str(rep)+'.csv'), index=False)

    EA = EquationAdjustor(initial_hidden_values=hidden_values,
                          hidden_weights=hidden_weights,
                          activation=np.tanh,
                          horizontal=horizontal,
                          initial_adjustment=initial_adjustment,
                          initial_parameter=initial_parameter, # should go in __init__. scaling will be 1
                          num_adjustments=num_adjustments,
                          fixed_adjustments=fixed_adjustments,
                          variant_string=variant_string)

    xopt, es = cma.fmin2(EA.cma_es_function, weights, sigma,
                         args=(rng, all_datasets, return_all_errors),
                         options={'maxfevals': function_evals,
                                  # 'ftarget': 1e-10,
                                  'tolfun': 0,
                                  # 'tolfunhist': 0,
                                  'seed': seed,
                                  'verb_log': 0,
                                  'timeout': timeout},
                         restarts=0)

    xopt = best[1]
    
    # testing = EA.cma_es_function(xopt, rng, all_datasets,
    #                                        return_all_errors=True)


    all_datasets['validation'] = datasets_validation_functions_consistent

    start_time = time.time()

    errors, fitnesses = EA.cma_es_function(xopt, rng, all_datasets, return_all_errors=True)

    test_time = time.time() - start_time

    if not debug_mode:

        test_computation = test_time*cycles_per_second

        with open(os.path.join(save_loc, 'test_computation_and_cycles_rep'+str(rep)+'_'+variant_string+'.txt'), mode='w') as f:
            f.write(str(test_computation)+' '+str(cycles_per_second))

    # save the best individual
    data = [list(xopt)]
    df = pd.DataFrame(data).transpose()
    df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(rep)+'_'+variant_string+'.csv'), header=['trained weights'])
                                                                              # 'untrained weights',
                                                                              # 'initial hidden values'])
    # save the best individual function validation info
    header = ['Target', 'Validation Fitness']

    table = [['Validation Function ' + str(i), f] for i, f in enumerate(fitnesses['validation'])]

    df = pd.DataFrame(table, columns=header)
    df.to_csv(os.path.join(save_loc, 'best_ind_validation_rep'+str(rep)+'_'+variant_string+'.csv'))

    # save the best individual function testing info
    header = ['Target', 'Adjustment', 'Test Error', 'CPU Time', 'Computation']

    table = []

    for target in errors['testing']:

        for i, (e, t) in enumerate(zip(errors['testing'][target]['errors'], errors['testing'][target]['cpu time'])):

            table.append([target, i, e, t, t*cycles_per_second])


    df = pd.DataFrame(table, columns=header)
    df.to_csv(os.path.join(save_loc, 'best_ind_testing_rep'+str(rep)+'_'+variant_string+'.csv'))


    # # save the best individual
    # data = [list(errors['training']), list(fitnesses['training']), list(errors['validation']), list(fitnesses['validation']), list(fitnesses['testing']), list(fitnesses['testing']), list(EA.errors), list(EA.cpu_time), list(fitnesses_test_function['testing']), list(xopt), list(hidden_weights.flatten()), list(hidden_values)]
    # df = pd.DataFrame(data).transpose()
    # df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(rep)+'.csv'), header=['train error',
    #                                                                           'train fitness',
    #                                                                           'validation error',
    #                                                                           'validation fitness',
    #                                                                           'test error',
    #                                                                           'test fitness',
    #                                                                           'test error on test function',
    #                                                                           'cpu time',
    #                                                                           'test fitness on test funciton',
    #                                                                           'trained weights',
    #                                                                           'untrained weights',
    #                                                                           'initial hidden values'])


def get_benchmark_datasets(rng, max_shift, horizontal):

    target_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                    'keijzer15', 'r1', 'r2', 'r3']

    target_strings = ['x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0])))',
                      'x[0] * (1 - x[0] * (2 - x[0] * (1 - x[0] * (1 - x[0] * (1 - x[0] * (2 - x[0]))))))',
                      'x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0]))))))))',
                      '(x[0] * x[1]) + np.sin((x[0] - 1) * (x[1] - 1))',
                      'x[0] ** 4 - x[0] ** 3 + (x[1] ** 2 / 2.0) - x[1]',
                      '6 * np.sin(x[0]) * np.cos(x[1])',
                      '8.0 / (2 + x[0] ** 2 + x[1] ** 2)',
                      '(x[0] ** 3 / 5.0) + (x[1] ** 3 / 2.0) - x[0] - x[1]',
                      '((x[0] + 1) ** 3) / (x[0] ** 2 - x[0] + 1)',
                      '(x[0] ** 5 - (3 * (x[0] ** 3)) + 1) / (x[0] ** 2 + 1)',
                      '(x[0] ** 6 + x[0] ** 5) / (x[0] ** 4 + x[0] ** 3 + x[0] ** 2 + x[0] + 1)']

    benchamrk_datasets = {}

    for i, (f_name, f_func) in enumerate(zip(target_names, target_strings)):
    
        shift = rng.uniform(0, max_shift)

        x_train = rng.uniform(-1, 1, 20)
        y_train = rng.uniform(-1, 1, 20)

        input_train = np.vstack((x_train, y_train))

        x_val = rng.uniform(-1, 1, 20)
        y_val = rng.uniform(-1, 1, 20)

        input_val = np.vstack((x_val, y_val))

        x_test = rng.uniform(-1, 1, 1000)
        y_test = rng.uniform(-1, 1, 1000)

        input_test = np.vstack((x_test, y_test))

        if horizontal:

            lambda_string = 'lambda x: '+f_func.replace('x[0]', 'np.add(x[0],'+str(shift)+')')

        else:

            lambda_string = 'lambda x: '+f_func + '+' +str(shift)

        f = eval(lambda_string)

        output_train = f(input_train)
        output_val = f(input_val)
        output_test = f(input_test)

        dataset = ((f_func,
                   np.vstack([output_train, input_train]).T,
                   np.vstack([output_val, input_val]).T,
                   np.vstack([output_test, input_test]).T),)

        benchamrk_datasets[f_name] = dataset

    return benchamrk_datasets


def get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target,
                                    depth, max_shift=50, horizontal=False, fixed_adjustments=False):
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
    datasets : list
        List of tuple containnig (base_function, dataset).
    """

    primitives = ['*', '+', '%', '-']
    terminals = ['#x', '#f']

    # Pick the number of input variables
    # between 1 and 6.
    num_vars = 1    #rng.choice(5)+1

    targets = []

    for _ in range(num_targets):

        t = GP.Individual(rng=rng, primitive_set=primitives, terminal_set=terminals, num_vars=num_vars,
                          depth=depth, method='grow')

        f = eval('lambda x: ' + t.convert_lisp_to_standard_for_function_creation())
        outputs = f(np.array([np.linspace(-10, 10, 1000)]).T)

        while 'x0' not in t.get_lisp_string() or len(np.unique(np.around(outputs,7))) == 1 or np.any([t.get_lisp_string() == ind.get_lisp_string() for ind in targets]) or t.get_lisp_string() == '(x0)':

            t = GP.Individual(rng=rng, primitive_set=primitives, terminal_set=terminals, num_vars=num_vars,
                              depth=depth, method='grow')

            f = eval('lambda x: ' + t.convert_lisp_to_standard_for_function_creation())
            outputs = f(np.array([np.linspace(-10, 10, 1000)]))

        targets.append(t)

    datasets = []

    for i, t in enumerate(targets):

        offset = np.zeros(num_base_function_per_target)

        if fixed_adjustments:
            rand_offset = rng.randint

        else:
            rand_offset = rng.uniform

        while 0. in offset:
            offset = rand_offset(-max_shift, max_shift, size=num_base_function_per_target)

        # base_file = os.path.join(os.environ['GP_DATA'], 'tree')

        for o in offset:

            base_function_string = t.convert_lisp_to_standard_for_function_creation()

            if horizontal:
                function_string = base_function_string.replace('x[0]', 'np.add(x[0],'+str(o)+')')

            else:
                function_string = base_function_string+'+'+str(o)

            function = eval('lambda x: '+function_string)

            # Make inputs
            x = np.array([rng.uniform(-1, 1, size=300) for _ in range(num_vars)]).T

            training_indices = rng.choice(300, size=100, replace=False)
            remaining_indices = [i for i in range(300) if i not in training_indices]
            validation_indices = rng.choice(remaining_indices, size=100, replace=False)
            testing_indices = np.array([i for i in range(300) if i not in training_indices and i not in validation_indices])

            x_training = x[training_indices]
            x_validation = x[validation_indices]
            x_testing = x[testing_indices]

            training_dataset = np.hstack((np.array([function(x_training.T)]).T, x_training))
            validation_dataset = np.hstack((np.array([function(x_validation.T)]).T, x_validation))
            testing_dataset = np.hstack((np.array([function(x_testing.T)]).T, x_testing))

            datasets.append((base_function_string, training_dataset, validation_dataset, testing_dataset))

    return datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # this will act as an index for rep_list (offset by 1 though)
    parser.add_argument('rep', help='Number of runs already performed', type=int)
    parser.add_argument('exp', help='Experiment number. Used in save location', type=int)
    parser.add_argument('-hs', '--horizontal_shift', help='If True, NN will be trained to do horizontal shifts',
                        action='store_true')
    parser.add_argument('-t', '--timeout', help='The number of seconds use for training based on clock speed.',
                        type=float, action='store')

    parser.add_argument('-d', '--debug_mode', help='Do not adjust timeout',
                        action='store_true')

    parser.add_argument('-gp', '--genetic_programming', help='Compare with GP',
                        action='store_true')

    parser.add_argument('-vs', '--variant_string', help='The string that describes NN options',
                        type=str)

    args = parser.parse_args()
    print(args)

    num_adjustments = 50

    if not args.horizontal_shift:
        max_shift = sum([1-k/num_adjustments for k in range(num_adjustments)])

    else:
        max_shift = 5

    benchamrk_datasets = get_benchmark_datasets(np.random.RandomState(args.rep+100*args.exp), max_shift, args.horizontal_shift)

    if args.genetic_programming:

        print('genetic programming')

        for function in benchamrk_datasets:

            rng = np.random.RandomState(args.rep+100*args.exp)

            path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(args.exp))

            # dataset is the training dataset and validation dataset
            dataset = [benchamrk_datasets[function][0][1], benchamrk_datasets[function][0][2]]
            test_data = benchamrk_datasets[function][0][3]

            if args.debug_mode:
                timeout = args.timeout
                cycles_per_second = 1.6*10**9
        
            else:
                timeout, cycles_per_second = get_computation_time(args.timeout, return_cycles_per_second=True)

            # get output_path, output_file
            output_path = os.path.join(path, 'gp', function)
            output_file = 'fitness_data_rep' + str(args.rep) + '.csv'
            
            params = {'T': timeout,
                      'cycles_per_second': cycles_per_second}

            gp = GP.GeneticProgrammingAfpo(rng=rng,
                                           pop_size=100,
                                           max_gens=30000,
                                           primitive_set=['*', '+', '%', '-'],
                                           terminal_set=['#x', '#f'],
                                           # this is not data, which is passed
                                           data=dataset,
                                           test_data=test_data,
                                           prob_mutate=1,
                                           prob_xover=0,
                                           num_vars=1,
                                           mutation_param=2,
                                           # parameters below
                                           **params)

            info = gp.run(rep=args.rep,
                          output_path=output_path,
                          output_file=output_file)

    else:
        print('equation adjuster')
        assert args.timeout is not None, 'Specify a time limit with -t or --timeout'

        train_equation_corrector(rep=args.rep, exp=args.exp, timeout=args.timeout,
                                 fixed_adjustments=False, horizontal=args.horizontal_shift,
                                 debug_mode=args.debug_mode, benchamrk_datasets=benchamrk_datasets,
                                 max_shift=max_shift, num_adjustments=num_adjustments,
                                 variant_string=args.variant_string)
