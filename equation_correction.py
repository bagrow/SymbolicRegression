import GeneticProgramming as GP
from GeneticProgramming.protected_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma

import argparse
import os

class EquationAdjustor:

    def __init__(self, initial_hidden_values, hidden_weights, activation, horizontal, datasets,
                 initial_adjustment, initial_parameter, num_adjustments, fixed_adjustments):
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
        datasets : list of tuples
            Each tuple contains the base function (as
            as string) and the dataset for that function
            with an offset (vertical or horizontal)
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

        self.initial_hidden_values = initial_hidden_values
        self.hidden_weights = hidden_weights
        self.activation = activation

        self.datasets = datasets

        self.initial_adjustment = initial_adjustment
        self.initial_parameter = initial_parameter
        self.num_adjustments = num_adjustments

        self.fixed_adjustments = fixed_adjustments

        self.horizontal = horizontal

        if self.horizontal:
            self.adjust_function = lambda function_string, c: eval('lambda x: '+function_string.replace('x[0]', 'np.add(x[0],'+str(c)+')'))
        
        else:   # vertical shift
            self.adjust_function = lambda function_string, c: eval('lambda x: '+function_string+'+'+str(c))


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

        return output[:-1]


    def evaluate_corrector_neural_network(self, w, error, signed_error, prev_error, prev_index):
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
            input = [prev_error, error, *np.eye(2)[int(prev_index)]]

        else:
            input = [signed_error]

        # Take one node for the signed error.
        input_weights = w[:num_hidden*len(input)].reshape((len(input), num_hidden))

        # Get recurrent weights
        hidden_weights = self.hidden_weights

        # Get the output weights. Output is one of three nodes.
        output_weights = w[num_hidden*len(input):].reshape((num_hidden, 3))

        output = self.get_value(input_weights, hidden_weights, output_weights, input)

        index = np.argmax(output)

        return index


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

        prev_error = self.errors[-2] if len(self.errors) > 1 else 0.

        self.index = self.evaluate_corrector_neural_network(w, error=self.errors[-1]/self.errors[0],
                                                            signed_error=self.signed_error/self.errors[0],
                                                            prev_error=prev_error/self.errors[0], 
                                                            prev_index=self.index)

        # if self.errors[-1] < prev_error:
        #     pass

        # elif self.errors[-1] > prev_error:
        #     self.index = (self.index + 1) % 2

        # else:
        #     self.index = 0

        if self.index == 0:
            self.parameter += self.adjustment

        elif self.index == 1:
            self.parameter -= self.adjustment

        elif self.index == 2:
            pass # keep self.parameter the same

        else:
            print('Unspecified option. index =', index)
            exit()

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

        fitness = self.errors[-1]/self.errors[0]

        return fitness


    def cma_es_function(self, w, rng, return_all_errors=False):
        """Function that is passed to CMA-ES."""

        fitnesses = []
        errors = []

        for function_string, dataset in self.datasets:

            self.reinitialize(function_string, dataset)

            fitness = self.run_equation_corrector(function_string, dataset, w)

            fitnesses.append(fitness)
            errors.append(self.errors[-1])

        if return_all_errors:
            return fitnesses

        else:
            return np.mean(fitnesses)


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


def train_equation_corrector(rep, fixed_adjustments, horizontal):

    rng = np.random.RandomState(rep)

    hidden_values = rng.uniform(-1, 1, size=10)
    hidden_weights = rng.uniform(-1, 1, size=(len(hidden_values), len(hidden_values)))
    num_input = 4 if horizontal else 1
    num_output = 3

    weights = rng.uniform(-1, 1, size=num_input*len(hidden_values)+len(hidden_values)*num_output)
    
    # get data
    num_targets = 50
    num_base_function_per_target = 2
    depth = 3

    sigma = 2.
    function_evals = float('inf')
    seed = args.rep + 1
    # timeout = get_computation_time(18000)
    timeout = 18000

    return_all_errors = False

    # max_offset = sum([1-k/num_iterations for k in range(num_iterations)])
    max_offset = 5

    datasets = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth, max_offset, horizontal, fixed_adjustments)
    print('test error below')
    datasets_test = get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target, depth, max_offset, horizontal, fixed_adjustments)

    EA = EquationAdjustor(initial_hidden_values=hidden_values,
                          hidden_weights=hidden_weights,
                          activation=np.tanh,
                          horizontal=horizontal,
                          datasets=datasets,
                          initial_adjustment=1.,
                          initial_parameter=0., # should go in __init__. scaling will be 1
                          num_adjustments=50,
                          fixed_adjustments=fixed_adjustments)

    xopt, es = cma.fmin2(EA.cma_es_function, weights, sigma,
                     args=(rng, return_all_errors),
                     options={'maxfevals': function_evals,
                              # 'ftarget': 1e-10,
                              'tolfun': 0,
                              # 'tolfunhist': 0,
                              'seed': seed,
                              'verb_log': 0,
                              'timeout': timeout},
                     restarts=0)

    test_errors = EA.cma_es_function(xopt, rng, return_all_errors=True)

    save_loc = os.path.join(os.environ['GP_DATA'], 'equation_adjuster')


    # save the best individual
    # ['test error', 'trained weights', 'untrained weights', 'initial hidden values']
    data = [list(test_errors), list(xopt), list(hidden_weights.flatten()), list(hidden_values)]
    df = pd.DataFrame(data).transpose()
    df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(rep)+'.csv'), header=['test error', 'trained weights', 'untrained weights', 'initial hidden values'])

    # plt.hist(test_errors)
    # plt.ylabel('Frequency')
    # plt.xlabel('Test Error')
    # plt.savefig(os.path.join(save_loc, 'figures', 'hist_final_errors_rep'+str(args.rep)+'.pdf'))


def get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target,
                                    depth, max_offset=50, horizontal=False, fixed_adjustments=False):
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

        offset = [0., 0.]

        if fixed_adjustments:
            rand_offset = rng.randint

        else:
            rand_offset = rng.uniform

        while 0. in offset:
            offset = rand_offset(-max_offset, max_offset, size=num_base_function_per_target)

        # base_file = os.path.join(os.environ['GP_DATA'], 'tree')

        for o in offset:

            base_function_string = t.convert_lisp_to_standard_for_function_creation()

            if horizontal:
                function_string = base_function_string.replace('x[0]', 'np.add(x[0],'+str(o)+')')

            else:
                function_string = base_function_string+'+'+str(o)

            function = eval('lambda x: '+function_string)

            # Make inputs
            x = np.array([rng.uniform(-1, 1, size=100) for _ in range(num_vars)]).T

            dataset = np.hstack((np.array([function(x.T)]).T, x))

            datasets.append((base_function_string, dataset))

    return datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # this will act as an index for rep_list (offset by 1 though)
    parser.add_argument('rep', help='Number of runs already performed', type=int)
    parser.add_argument('-hs', '--horizontal_shift', help='If True, NN will be trained to do horizontal shifts',
                        action='store_true')

    args = parser.parse_args()

    train_equation_corrector(args.rep, fixed_adjustments=False, horizontal=args.horizontal_shift)
