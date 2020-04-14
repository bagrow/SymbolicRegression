from .common_functions import union, get_function
from .Individual import Individual
from .consts import *
from .protected_functions import *

import numpy as np


class IndividualManyTargetData(Individual):


    def evaluate_fitness(self, data, attempts=40, compute_val_error=True):
        """Calculate error for training data. If computing
        constants, try attempts number of times. Each time
        pick some random inital guess for the non-linear
        least squares algorithm.

        Note: using computed constants slows this method
        down!

        Parameters
        ----------
        data : list of np.array
            The list contains the datasets (training and testing),
            which are np.arrays. At the top layer, the
            list is split into training and validation datasets.
            Next, into the actual data with output followed by
            each input. That is, a row of data is of the form
            y, x0, x1, ...
        attemps : int (default=40)
            Number of times to compute the consants using different
            guesses. The best constants are used. This argument only
            has an effect if computed constants are being used.
        compute_val_error : bool (default=True)
            If True, compute the validation error. The dataset for
            this is stored in data[1].
        """

        self.fitness[0] = 0

        if compute_val_error:
            x_data_val = data[1][:, 1:].T

                        # if x is the wrong size, adjust
            num_x_input = len(x_data_val)

            if num_x_input < self.num_vars:
                x_adjusted_val = np.zeros((self.num_vars, len(x_data_val[0])))
                x_adjusted_val[:num_x_input, :] = x_data_val.copy()

            else:
                x_adjusted_val = x_data_val.copy()

            y_data_val = data[1][:, 0]

        f_string = self.convert_lisp_to_standard_for_function_creation()

        self.f = get_function(f_string)

        error = lambda x, y, f=self.f: np.sqrt(np.mean(np.power(f(x) - y, 2)))

        for train_dataset in data[0]:

            x_data = train_dataset[:, 1:].T

            # if x is the wrong size, adjust
            num_x_input = len(x_data)

            if num_x_input < self.num_vars:
                x_adjusted = np.zeros((self.num_vars, len(x_data[0])))
                x_adjusted[:num_x_input, :] = x_data.copy()

            else:
                x_adjusted = x_data.copy()

            y_data = train_dataset[:, 0]

            self.fitness[0] += error(x_adjusted, y_data)

        self.fitness[0] = self.fitness[0] / len(data[0])

        if compute_val_error:
            self.validation_fitness = error(x_adjusted_val, y_data_val)

    def evaluate_test_points(self, data):
        """Calculate error for testing data. This method assumes that self.f
        exists, which is usually created by running
        Individual.evaluate_fitness

        Parameters
        ----------
        data : np.array
            An np.array of 2D np.arrays. At the top layer, the
            list is split into training and validation datasets.
            Next, into the actual data with output followed by
            each input. That is, a row of data is of the form
            y, x0, x1, ...
        """

        x_data = data[:, 1:].T
        y_data = data[:, 0]

        # if x is the wrong size, adjust
        num_x_input = len(x_data)

        if num_x_input < self.num_vars:
            x_adjusted = np.zeros((self.num_vars, len(x_data[0])))
            x_adjusted[:num_x_input, :] = x_data.copy()

        else:
            x_adjusted = x_data.copy()

        try:

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(self.c, x_data) - y_data, 2)))

        except (TypeError, AttributeError) as e:   # if no var self.c

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(x_adjusted) - y_data, 2)))

        return self.testing_fitness


    def get_number_of_operations_in_tree_eval(self, datasets):
        """Modified from parent class to account for eval on multiple datasets
        and averaging over them. Each non-leaf node is an operation, so
        the number of operations in a single evaluation
        of a tree is equal to the number of non-leaf
        nodes. Then, RMSE is done.

        Parameters
        ----------
        datasets : list
            The training and validation datasets

        Returns
        -------
        num_non-leaves : int
            The number of non-leaf nodes (number of floating
            point operations performed per evaluation)
        """

        num_leaves, num_nodes = self.get_num_leaves(num_nodes=True)
        num_nonleaves = num_nodes - num_leaves

        num_ops_per_eval = num_nonleaves

        num_training_data_points = sum([len(d) for d in datasets[0]])

        num_data_points = num_training_data_points + len(datasets[1])

        num_ops_per_RMSE = 3*num_data_points + 1

        num_ops_per_average = len(datasets[0])

        return num_ops_per_RMSE + num_data_points*num_ops_per_eval
