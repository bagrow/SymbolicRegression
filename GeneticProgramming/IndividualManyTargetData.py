from .common_functions import union, get_function
from .Individual import Individual
from .consts import *
from .protected_functions import *
from .errors import RSE, RMSE

import numpy as np


class IndividualManyTargetData(Individual):


    def get_error(self, dataset_list, error):

        errors = []

        for dataset in dataset_list:

            x_data = dataset[:, 1:].T

            # if x is the wrong size, adjust
            num_x_input = len(x_data)

            if num_x_input < self.num_vars:
                x_adjusted = np.zeros((self.num_vars, len(x_data[0])))
                x_adjusted[:num_x_input, :] = x_data.copy()

            else:
                x_adjusted = x_data.copy()

            y_data = dataset[:, 0]

            errors.append(error(x=x_adjusted.T, y=y_data[:,None], f=self.f))

        return np.mean(errors)


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

        f_string = self.convert_lisp_to_standard_for_function_creation()

        self.f = get_function(f_string)

        error = RSME

        self.fitness[0] = self.get_error(dataset_list=data[0], error=error)

        if compute_val_error:

            self.validation_fitness = self.get_error(dataset_list=data[1], error=error)


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


    def get_effort_tree_eval(self, datasets):
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

        effort_per_eval = num_nonleaves

        num_training_data_points = sum([len(d) for d in datasets[0]])
        num_validation_data_points = sum([len(d) for d in datasets[1]])

        num_data_points = num_training_data_points + num_validation_data_points

        effort_from_RMSE = 3*num_data_points + 1

        effort_from_average = len(datasets[0])

        return effort_from_RMSE + effort_from_average + num_data_points*effort_per_eval
