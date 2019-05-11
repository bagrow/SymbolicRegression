import functions.common_functions as cf
from classes.tree import Tree
from consts import *
from protected_functions import *  # these are the protected functions

import scipy.optimize
import numpy as np

import copy
from itertools import count


class Individual(Tree):
    """The class is a individual in a symbolic regression algorithm."""


    _ids = count(0)
    evaluation_count = count(0)

    def __init__(self, rng, primitive_set, terminal_set, num_vars=1,
                 age=0, max_depth=6, depth=None, tree=None, method=None,
                 **params):
        """Initialize Individual

        Parameters:
            primitive_set: (required) a list of all allowed functions.

            terminal_set: (required) a list of all allowed terminals (constants, variables).

            age: (default = 0) A non-negative integer that describes the amount of time for which the individual
            has been evolved. As describe in AFPO.

            data: A list of 2D np.arrays. At the top layer, the list is split into training, validation, and testing
            data. Next, into the actual data with output followed by each input.

            max_depth: (default = 6) A non-negative integer that limits the depth of the tree.

            depth: A non-negative integer that determines the initial max depth of the tree. This
            parameter should only be used in conjunction with method. Otherwise, it is not used.

            tree: Pass a list of list to represent the tree (self.tree).

            method: A string such as 'grow' or 'full' which determines the method for creating the tree.
            If left blank, one of the previously mentioned methods is selected at random."""

        # Run parent classes __init__, but possibly don't specify the tree yet
        Tree.__init__(self, tree=tree, num_vars=num_vars, rng=rng, **params)

        # training, validation, testing, size
        self.fitness = np.array([0., 0.])

        # # This variable differs from fitness error because it will have the error at each data point.
        # self.error_data = []

        # fitness (that does not effect other fitness) during symbolic regression
        self.validation_fitness = 0

        # fitness on data after symbolic regression finishes
        self.testing_fitness = 0

        # # number of times error (at a single data point) < 10^-3
        # self.hits = 0

        # We don't want any repeated elements in these lists,
        # but using rng.choice on set does not always produce
        # the same results, even with the same seed. So, we
        # use lists
        self.P = primitive_set
        self.T = terminal_set
        self.C = cf.union(self.P, self.T)

        self.max_depth = max_depth
        self.age = age

        # For tracking lineages
        self.parentID = None
        self.id = next(self._ids)

        if depth is None:

            depth = max_depth

        # Figure out which method to use to create tree. If none specified pick one at random.
        if tree is not None:

            pass    # already did this part by calling parent __init__() above

        else:

            if method is None:

                full = self.rng.choice((False, True))    # If full is false (0) use grow method

                if full:

                    self.tree = self.generate_random_individual_full(depth)

                else:

                    self.tree = self.generate_random_individual_grow(depth)

            elif method == 'grow':

                self.tree = self.generate_random_individual_grow(depth)

            elif method == 'full':

                self.tree = self.generate_random_individual_full(depth)

            else:

                print("Error:", method, "is an unknown method.")

            self.apply_rules_to_tree()

    # ---------------------------------------------------------------------- #
    #                       Tree Creation Functions
    # ---------------------------------------------------------------------- #

    def generate_random_individual_full(self, max_depth):
        """Generate a tree using the full method. This method creates each node from the function set until
        max depth is reach at which point the node is selected from the terminal set. A node created with this
        method will be exactly depth max_depth (in all branches) every time.

        Parameters:
            max_depth: This parameter determine the max depth of
            the node returned by this function."""

        tree = []
        current_indices = None
        current_depth = 0

        while current_depth <= max_depth:

            if current_indices is None:

                current_indices = []

                # decide if a terminal or primitive is to be placed
                if current_depth < max_depth:

                    primitive = self.rng.choice(self.P)

                    tree.append(primitive)

                    # give child nodes a place to exist
                    for i in range(required_children_for_function[primitive]):

                        tree.append([])
                        current_indices.append([i+1])   # point to these child nodes

                else:

                    terminal = self.rng.choice(self.T)

                    tree.append(terminal)

            else:

                new_current_indices = []

                for index_list in current_indices:

                    subtree = tree

                    # select the subtree
                    for index in index_list:

                        subtree = subtree[index]

                    # decide if a terminal or primitive is to be placed
                    if current_depth < max_depth:

                        primitive = self.rng.choice(self.P)

                        subtree.append(primitive)

                        # append place for child nodes
                        for i in range(required_children_for_function[primitive]):

                            subtree.append([])

                            # point the child nodes for next iteration
                            new_current_indices = new_current_indices + [index_list + [i + 1]]

                    else:

                        terminal = self.rng.choice(self.T)

                        subtree.append(terminal)

                # update indices
                current_indices = new_current_indices

            current_depth += 1

        return tree


    def generate_random_individual_grow(self, max_depth):
        """Generated a tree using the grow method. This method is can be any depth up to max_depth.
        Each node can be of any type, unless at max depth, in which case, the node
        must be a terminal node.

        Parameters:
            max_depth: This parameter determine the max depth of the node returned by this function."""

        tree = []
        current_indices = None
        current_depth = 0

        while current_depth <= max_depth:

            if current_indices is None:

                current_indices = []

                # decide if a terminal is required
                if current_depth < max_depth:

                    primitive = self.rng.choice(self.C)

                    tree.append(primitive)

                    try:

                        for i in range(required_children_for_function[primitive]):

                            tree.append([])
                            current_indices.append([i+1])

                    except KeyError:

                        # primitive is actually a terminal
                        pass

                else:   # terminal is required

                    terminal = self.rng.choice(self.T)

                    tree.append(terminal)

            else:

                new_current_indices = []

                for index_list in current_indices:

                    subtree = tree

                    # select the subtree
                    for index in index_list:

                        subtree = subtree[index]

                    if current_depth < max_depth:

                        primitive = self.rng.choice(self.C)

                        subtree.append(primitive)

                        try:

                            # append place for child nodes
                            for i in range(required_children_for_function[primitive]):

                                subtree.append([])
                                new_current_indices = new_current_indices + [index_list + [i + 1]]

                        except KeyError:

                            # picked a terminal
                            pass

                    else:

                        terminal = self.rng.choice(self.T)

                        subtree.append(terminal)

                # update indices
                current_indices = new_current_indices

            current_depth += 1

        return tree

    # ----------------------------------------------------------------------------- #
    #                       Multi-Objective Functions
    # ----------------------------------------------------------------------------- #

    def dominates(self, ind):
        """Check if self dominates ind (individual). self must be better
        in at least one objective and at least as good in all objectives."""

        # if fitness values ar the same
        if self == ind:

            return self.id < ind.id

        # otherwise
        else:

            return np.any(self.fitness < ind.fitness) and np.all(self.fitness <= ind.fitness)


    def neither_dominates(self, ind):
        """Check if self does not dominate ind (individual) and vice versa."""

        # if fitness values ar the same
        if self == ind:

            return self.id == ind.id

        elif np.any(self.fitness < ind.fitness) and np.all(self.fitness <= ind.fitness):

            return False

        elif np.any(self.fitness > ind.fitness) and np.all(self.fitness >= ind.fitness):

            return False

        else:

            return True

    # ----------------------------------------------------------------------------- #
    #                                  Mutations
    # ----------------------------------------------------------------------------- #

    def get_possible_mutations_no_values(self, new_tree, subtree, child_index_list, mutation_param):
        """Given node in new_tree determine which mutation are possible. Does not require the
        use of values of each node.

        Parameters:
            new_tree: A copy of the self.tree

            subtree: selected subtree of new_tree

            mutation_param: mutation parameter describing the depth of tree
            to create on mutation (node_replacement)

            child_index_list: list of child indices that describes the location of node in tree.
        """

        mut_list = [self.node_replacement]
        mut_param = [(new_tree, child_index_list, mutation_param)]

        # Don't include any mutation involving constants if no constants are used.
        if ('#f' in self.T or '#i' in self.T or '#c' in self.T) and self.is_constant(subtree):

            mut_list.append(self.constant_mutation)
            mut_param.append((new_tree, child_index_list))

        # Don't include any mutation involving variables if a variable is not selected.
        if '#x' in self.T and self.is_variable(subtree):

            mut_list.append(self.variable_mutation)
            mut_param.append((new_tree, child_index_list))

        return (mut_list, mut_param)


    def mutate(self, mutation_param):
        """Pick a random node in the tree and then pick a mutation based on that particular node."""

        # Get list of all node in tree (individual).
        node_list = self.get_node_list()

        # Choose one node for the mutation location
        index = self.rng.choice(len(node_list))
        child_index_list = list(node_list[index])

        # Make a new tree that is currently identical to the old one.
        new_tree = self.__class__(rng=self.rng, primitive_set=self.P, terminal_set=self.T,
                                  num_vars=self.num_vars, age=self.age, max_depth=self.max_depth,
                                  tree=copy.deepcopy(self.tree), **self.params)

        # Select the subtree in new tree.
        subtree = new_tree.select_subtree(child_index_list=child_index_list)

        # Get list of possible mutation that could be applied at node.
        mut_list, mut_param = self.get_possible_mutations_no_values(new_tree=new_tree,
                                                                    subtree=subtree,
                                                                    child_index_list=child_index_list,
                                                                    mutation_param=mutation_param)

        # Select a mutation at random (uniformly) to apply.
        index = self.rng.choice(len(mut_list))

        # Mutate the individual.
        mutated_ind = mut_list[index](*mut_param[index])

        mutated_ind.apply_rules_to_tree()

        return mutated_ind


    def node_replacement(self, root_node, child_index_list, mutation_param=6):
        """Create new subtree and place it at choice_list.

        Parameters:
            root_node: The node that child_index_list considers to be the root.

            child_index_list: A list of child indices specifying the location to place
            the new subtree.

            mutation_param: specifies the max depth of the subtree."""

        if child_index_list == ['']:

            depth = 0

        else:

            depth = len(child_index_list)

        new_subtree = self.generate_random_individual_grow(min(mutation_param, self.max_depth - depth))

        root_node.set_subtree(new_subtree, child_index_list)

        return root_node


    def constant_mutation(self, root_node, child_index_list):
        """Perturb a constant node.

        Parameters:
            root_node: The node that child_index_list considers to be the root.

            child_index_list: A list of child indices specifying the location for
            the mutation to take place."""

        leaf = root_node.select_subtree(child_index_list)

        if '#c' in self.T:

            leaf[0] = 'c'+str(self.rng.choice(10))

        else:

            sigma = np.abs(leaf[0])

            leaf[0] = self.rng.normal(leaf[0], sigma)

        return root_node


    def variable_mutation(self, root_node, child_index_list):
        """Change variable to another variable.

        Parameters:
         root_node: The node that child_index_list considers to be the root.

            child_index_list: A list of child indices specifying the location for
            the mutation to take place."""

        leaf = root_node.select_subtree(child_index_list)

        if '#x' in self.T:

            leaf[0] = 'x'+str(self.rng.choice(self.num_vars))

        return root_node

    # --------------------------------------------------------------- #
    #                          Compute Error
    # --------------------------------------------------------------- #

    def evaluate_individual_error(self, data, is_non_dominated=False):
        """Evaluated error if the is_non_dominated parameter is
        False.

        Since the constant computations are not perfect, it is
        possible to increase the error in recomputing the constants of
        the exact same tree. So, save the previous error (and constants
        and function) and keep the best error after recomputing constants.
        Thus, the longer an individual exists, the more likely the optimal
        constants have been found. Is this unfair to young individuals? Perhaps,
        their youngness counteracts this?"""

        if not is_non_dominated:

            if hasattr(self, 'c'):

                old_error = self.fitness[0]
                old_c = self.c
                old_f = self.f

                self.evaluate_fitness_fast(data, attempts=1)

                if old_error < self.fitness[0]:

                    self.fitness[0] = old_error
                    self.c = old_c
                    self.f = old_f

            else:

                self.evaluate_fitness_fast(data)


    def evaluate_fitness_fast(self, data, attempts=40):
        """Calculate error for training data (0)"""

        self.fitness[0] = 0

        x_data = data[0, :, 1:].T
        x_data_val = data[1, :, 1:].T
        y_data = data[0, :, 0]
        y_data_val = data[1, :, 0]

        f_string = self.convert_lisp_to_standard_for_function_creation()

        if '#c' in self.T:

            are_consts = 'c[' in f_string
            self.f = cf.get_function(f_string, const=are_consts)

            if are_consts:

                residuals = lambda c, x, y, f=self.f: f(c, x) - y
                error = lambda c, x, y, f=self.f: np.sqrt(np.mean(np.power(residuals(c, x, y, f), 2)))

                history = []
                coeff_guess = self.rng.normal(0, 10, size=(10, attempts))  # parameter

                for i in range(attempts):  # parameter

                    res_lsq = scipy.optimize.least_squares(residuals,
                                                           x0=coeff_guess[:, i],
                                                           args=(x_data, y_data),
                                                           max_nfev=50,  # parameter
                                                           method='lm',
                                                           loss='linear')

                    history.append((res_lsq.x, res_lsq.cost))

                # sort by cost to find best const vals
                history.sort(key=lambda x: x[1])
                self.c = history[0][0]

                self.fitness[0] = error(self.c, x_data, y_data)
                self.validation_fitness = error(self.c, x_data_val, y_data_val)

            else:

                error = lambda x, y, f=self.f: np.sqrt(np.mean(np.power(f(x) - y, 2)))

                self.fitness[0] = error(x_data, y_data)
                self.validation_fitness = error(x_data_val, y_data_val)

        else:

            self.f = cf.get_function(f_string)
            print(f_string)

            error = lambda x, y, f=self.f: np.sqrt(np.mean(np.power(f(x) - y, 2)))

            self.fitness[0] = error(x_data, y_data)
            self.validation_fitness = error(x_data_val, y_data_val)


    def evaluate_test_points_fast(self, data):
        """Calculate error for testing data"""

        x_data = data[:, 1:].T
        y_data = data[:, 0]

        try:

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(self.c, x_data) - y_data, 2)))

        except (TypeError, AttributeError) as e:   # if no var self.c

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(x_data) - y_data, 2)))

        return self.testing_fitness


    def __eq__(self, other):
        """In terms of the pareto front. Two individuals are
        considered identical if their fitness (in all objectives)
        are equal."""

        return round(self.fitness[0], 7) == round(other.fitness[0], 7) and int(self.fitness[1]) == int(other.fitness[1])
