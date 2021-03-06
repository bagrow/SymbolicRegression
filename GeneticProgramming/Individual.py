from .common_functions import union, get_function
from .Tree import Tree
from .consts import *
from .protected_functions import *

import scipy.optimize
import numpy as np

import copy
from itertools import count


class Individual(Tree):
    """This class is a individual in a symbolic
    regression algorithm. It inherits the Tree
    class. In this class, fitness and error are
    synonimous."""

    # The following class variables are used
    # to give each individual a unique id.
    _ids = count(0)
    evaluation_count = count(0)

    def __init__(self, rng, primitive_set, terminal_set, num_vars=1,
                 age=0, max_depth=6, depth=None, tree=None,
                 method=None, **params):
        """Initialize Individual

        Parameters
        ----------
        rng : random number generator
            For example let rng=np.random.RandomState(0)
        primitive_set : list
            A list of all primitive (operators/functions)
            that may be used in trees.
        terminal_set: list
            A list of all allowed terminals (constants, variables).
        num_vars : int (default=1)
            The number of input variables to use. This must be
            specified if more than one input variable is necessary.
        age : int (default=0)
            A non-negative integer that describes the number of
            generations for which the individual has been evolved.
            As describe in AFPO.
        max_depth : int (default=6)
            A non-negative integer that limits the depth of the
            tree.
        depth : int (optional)
            A non-negative integer that determines the initial
            max depth of the tree. This parameter should only
            be used in conjunction with argument called method.
            Otherwise, it is not used or assumed to be max_depth.
        tree : list (of lists, optional)
            Pass a list of list to represent the tree
            like Tree.tree.
        method : str
            A string such as 'grow' or 'full' which determines
            the method for creating the tree. If left blank,
            one of the previously mentioned methods is selected
            at random.
        """

        # Run parent classes __init__, but possibly don't
        # specify the tree yet
        Tree.__init__(self,
                      tree=tree,
                      num_vars=num_vars,
                      rng=rng,
                      **params)

        self.fitness = np.zeros(2)

        # Order is error, age
        if hasattr(self.params, 'AFSPO'):

            if self.params['AFSPO']:
                self.fitness = np.zeros(3)

        if not hasattr(self.params, 'IA'):
            self.params['IA'] = False

        # fitness (that does not effect other fitness)
        # during symbolic regression
        self.validation_fitness = 0

        # fitness on data after symbolic regression finishes
        self.testing_fitness = 0

        # We don't want any repeated elements in these lists,
        # but using rng.choice on set does not always produce
        # the same results, even with the same seed. So, we
        # use lists
        self.P = primitive_set
        self.T = terminal_set
        self.C = union(self.P, self.T)

        self.max_depth = max_depth
        self.age = age

        # For tracking lineages
        self.parentID = None
        self.id = next(self._ids)

        if depth is None:

            depth = max_depth

        # Figure out which method to use to create tree.
        # If none specified pick one at random.
        if tree is not None:

            # already did this part by calling
            # parent __init__() above
            pass

        else:

            if method is None:

                # If full is false, use grow method
                full = self.rng.choice((False, True))

                if full:

                    self.tree = self.generate_individual_full(depth)

                else:

                    self.tree = self.generate_individual_grow(depth)

            elif method == 'grow':

                self.tree = self.generate_individual_grow(depth)

            elif method == 'full':

                self.tree = self.generate_individual_full(depth)

            else:

                print("Error:", method, "is an unknown method.")

            self.apply_rules_to_tree()

    # ---------------------------------------------------------------------- #
    #                       Tree Creation Functions
    # ---------------------------------------------------------------------- #

    def generate_individual_full(self, max_depth):
        """Generate a tree using the full method. This method
        creates each node from the primitive set until max
        depth is reached at which point the node is selected
        from the terminal set. A tree created with this
        method will be exactly depth max_depth (in all branches)
        every time.

        Parameters
        ----------
        max_depth : int
            This parameter determine the max depth of
            the tree returned by this function. The tree
            generated by this method will have depth of
            exactly max_depth.

        Examples
        --------
        >>> I = GP.Individual(np.random.RandomState(0),
                              primitive_set=['*', '+', '-'],
                              terminal_set=['#x'])
        >>> I.tree = I.generate_individual_full(3)
        >>> I.get_lisp_string()
        (+ (* (+ (#x) (#x)) (- (#x) (#x))) (+ (* (#x) (#x)) (- (#x) (#x))))
        """

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
                    for i in range(required_children[primitive]):

                        tree.append([])
                        current_indices.append([i+1])

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
                        for i in range(required_children[primitive]):

                            subtree.append([])

                            # point to the child nodes for next
                            # iteration
                            new_current_indices = new_current_indices + [index_list + [i + 1]]

                    else:

                        terminal = self.rng.choice(self.T)

                        subtree.append(terminal)

                # update indices
                current_indices = new_current_indices

            current_depth += 1

        return tree


    def generate_individual_grow(self, max_depth):
        """Generated a tree using the grow method. This method
        is can be any depth up to max_depth. Each node can be
        of any type, unless at max depth, in which case,
        the node must be a terminal node.

        Parameters
        ----------
        max_depth : int
            This parameter determine the max depth of
            the node returned by this function. The
            tree generated by this method may have depth
            less than max_depth.

        Examples
        --------
        >>> I = GP.Individual(np.random.RandomState(0),
                              primitive_set=['*', '+', '-'],
                              terminal_set=['#x'])
        >>> I.tree = I.generate_individual_grow(3)
        >>> I.get_lisp_string()
        (+ (* (#x) (#x)) (#x))
        """

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

                        for i in range(required_children[primitive]):

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
                            for i in range(required_children[primitive]):

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


    def generate_individual_ptc2(self, max_size, pdist, max_depth=None):
        """Probabilistic Tree-Creation 2 (ptc2).

        Luke, S. (2000). Two fast tree-creation algorithms for genetic
        programming. IEEE Transactions on Evolutionary Computation,
        4(3), 274-283.

        This algorithm is designed to generate a tree of a size between
        1 and max_size (or bigger by the largest number of inputs to any
        primitive currently in use.

        Parameters
        ----------
        max_size : int
            The maximum number of nodes of the tree desired. The
            trees may be a bit bigger than this.
        pdist : list
            List of probabilities (summing to 1) that determine
            the likelihood of selecting 1, 2, ..., max_size as
            the desired size for this specific run of the function.
            This list must have length max_size.
        max_depth : int (default=self.max_depth)
            The maximum depth of tree to be generated.

        Examples
        --------
        The following example will actually generate the individual
        twice: once with the grow or full method, which is chosen
        at random by the Individual constructor and for a second
        time when ptc2 is called. Perhaps I should make a flag
        to not generate a tree when Individual is called.

        >>> I = GP.Individual(np.random.RandomState(0),
                              primitive_set=['*', '+', '-'],
                              terminal_set=['#x'])
        >>> I.generate_individual_ptc2(5, [1/5.]*5)
        >>> I.get_lisp_string()
        (+ (x0) (x0))
        """

        max_depth = self.max_depth if max_depth is None else max_depth

        assert self.max_depth >= max_depth, 'ERROR in generate_individual_ptc2: '
        'max depth for sepcific tree is larger than overall max_depth'

        assert max_size == len(pdist), 'ERROR in generate_individual_ptc2: '
        'the length of pdist is not the same as max_size.'

        if self.tree is None:

            self.tree = []

        size = self.rng.choice(max_size, p=pdist)

        # If desired size is 1, pick  terminal
        if size == 1:
            self.tree = [self.rng.choice(self.T)]

        else:
            self.tree = [self.rng.choice(self.P)]

            d = 1   # depth so far (if terminals are added)
            s = 1   # size so far
            missing_nodes = []

            for i in range(required_children[self.tree[0]]):
                missing_nodes.append(((i,), d))
                self.tree.append([None])    # put space for children

            # Until the size is big enough, add primitives and record
            # the location of necessary child to add for a functional
            # tree.
            while size > len(missing_nodes) + s and len(missing_nodes) > 0:

                index = self.rng.choice(len(missing_nodes))
                loc, d = missing_nodes[index]

                if d == 0:
                    subtree = [self.rng.choice(self.T)]
                    self.set_subtree(new_subtree=subtree,
                                     child_indices=loc)
                    s += 1

                else:
                    subtree = [self.rng.choice(self.P)]

                    # get locations of children of subtree
                    # add these to missing_nodes
                    for i in range(required_children[subtree[0]]):
                        missing_nodes.append(((*loc, i), d+1))
                        subtree.append([None])

                    self.set_subtree(new_subtree=subtree,
                                     child_indices=loc)
                    s += 1

                del missing_nodes[index]

            # Fill in the missing nodes with terminals
            while len(missing_nodes) > 0:

                index = self.rng.choice(len(missing_nodes))
                loc, d = missing_nodes[index]

                subtree = [self.rng.choice(self.T)]
                self.set_subtree(new_subtree=subtree,
                                 child_indices=loc)

                del missing_nodes[index]

        self.apply_rules_to_tree()


    # ----------------------------------------------------------------------------- #
    #                       Multi-Objective Functions
    # ----------------------------------------------------------------------------- #

    def dominates(self, ind):
        """Check if self dominates ind (individual). self must be better
        in at least one objective and at least as good in all objectives.

        Parameters
        ----------
        ind : Individual
            Individual to compare with self.

        Returns
        -------
        True if self dominates ind, otherwise false.

        Example
        -------
        >>> I1 = I2 = GP.Individual(np.random.RandomState(0),
                                    primitive_set=['*', '+', '-'],
                                    terminal_set=['#x'])
        >>> I1.fitness = np.array([1., 4.])
        >>> I2.fitness = np.array([1., 5.])
        >>> I1.dominates(I2)
        True
        """

        # if fitness values are the same
        if self == ind:
            # take newer individual
            return self.id > ind.id

        # otherwise
        else:
            return np.any(self.fitness < ind.fitness) and np.all(self.fitness <= ind.fitness)

    # ----------------------------------------------------------------------------- #
    #                                  Mutations
    # ----------------------------------------------------------------------------- #

    def get_possible_mutations(self, new_tree, subtree, child_indices, mutation_param):
        """Given node in new_tree determine which mutations are possible.
        Does not require the use of values of each node.

        Parameters
        ----------
        new_tree : list (of lists)
            A copy of the self.tree
        subtree : list (of lists)
            Selected subtree of new_tree
        child_indices : iterable
            List of child indices that describes the location of node
            in tree.
        mutation_param : int
            Mutation parameter describing the max_depth of subtree
            to create on mutation (node_replacement).

        Returns
        -------
        mut_list : list (of functions)
            The list of possible mutation as functions to be
            called.
        mut_params : list (of tuples)
            The list (in same order as mut_list) with the parameters
            necessary to call the functions in mut_list
        """

        mut_list = [self.node_replacement]
        mut_param = [(new_tree, child_indices, mutation_param)]

        # Don't include any mutation involving constants if
        # no constants are used.
        if ('#f' in self.T or '#i' in self.T or '#c' in self.T) and self.is_constant(subtree):

            mut_list.append(self.constant_mutation)
            mut_param.append((new_tree, child_indices))

        # Don't include any mutation involving variables if
        # a variable is not selected.
        if '#x' in self.T and self.is_variable(subtree):

            mut_list.append(self.variable_mutation)
            mut_param.append((new_tree, child_indices))

        return (mut_list, mut_param)


    def mutate(self, mutation_param):
        """Pick a random node in the tree and then pick a mutation
        based on that particular node.

        Parameters
        ----------
        mutation_param : int
            Mutation parameter describing the max_depth of subtree
            to create on mutation (node_replacement).

        Returns
        -------
        mutated_ind : Individual
            The mutated version of self.
        """

        # Get list of all node in tree (individual).
        node_list = self.get_node_list()

        # Choose one node for the mutation location
        index = self.rng.choice(len(node_list))
        child_indices = node_list[index]

        # Make a new tree that is currently identical to the old one.
        new_tree = self.__class__(rng=self.rng, primitive_set=self.P,
                                  terminal_set=self.T, num_vars=self.num_vars,
                                  age=self.age, max_depth=self.max_depth,
                                  tree=copy.deepcopy(self.tree), **self.params)

        # Select the subtree in new tree.
        subtree = new_tree.select_subtree(child_indices=child_indices)

        # Get list of possible mutation that could be applied at node.
        mut_list, mut_param = self.get_possible_mutations(new_tree=new_tree,
                                                          subtree=subtree,
                                                          child_indices=child_indices,
                                                          mutation_param=mutation_param)

        # Select a mutation at random (uniformly) to apply.
        index = self.rng.choice(len(mut_list))

        # Mutate the individual.
        mutated_ind = mut_list[index](*mut_param[index])

        mutated_ind.apply_rules_to_tree()

        return mutated_ind


    def node_replacement(self, subtree, child_indices, mutation_param=6):
        """Create new subtree and place it at choice_list.

        Parameters
        ----------
        subtree : list (of lists)
            The subtree in self.tree to be replaced.
        child_indices : iterable
            List of child indices that describes the location of node
            in tree.
        mutation_param : int (default=6)
            Specifies the max depth of the subtree.

        Returns
        -------
        subtree : list (of lists)
            The subtree in self.tree to be replaced. It is now mutated.
        """

        if child_indices == [] or child_indices == ():
            depth = 0

        else:
            depth = len(child_indices)

        new_subtree = self.generate_individual_grow(min(mutation_param, self.max_depth - depth))

        subtree.set_subtree(new_subtree, child_indices)

        return subtree


    def constant_mutation(self, subtree, child_indices):
        """Perturb a constant node.

        Parameters
        ----------
        subtree : list (of lists)
            The subtree in self.tree to be replaced.
        child_indices : iterable
            List of child indices that describes the location of node
            in tree.
        """

        leaf = subtree.select_subtree(child_indices)

        if '#c' in self.T:
            leaf[0] = 'c'+str(self.rng.choice(10))

        else:
            sigma = np.abs(leaf[0])
            leaf[0] = self.rng.normal(leaf[0], sigma)

        return subtree


    def variable_mutation(self, subtree, child_indices):
        """Change variable to another variable.

        Parameters
        ----------
        subtree : list (of lists)
            The subtree in self.tree to be replaced.
        child_indices : iterable
            List of child indices that describes the location of node
            in tree.
        """

        leaf = subtree.select_subtree(child_indices)

        if '#x' in self.T:
            leaf[0] = 'x'+str(self.rng.choice(self.num_vars))

        return subtree

    # --------------------------------------------------------------- #
    #                          Compute Error
    # --------------------------------------------------------------- #

    def evaluate_individual_error(self, data):
        """Since the constant computations are not perfect, it is
        possible to increase the error in recomputing the constants of
        the exact same tree. So, save the previous error (and constants
        and function) and keep the best error after recomputing constants.
        Thus, the longer an individual exists, the more likely the optimal
        constants have been found. Is this unfair to young individuals?
        Perhaps, their youngness counteracts this? No of this applies if
        ephemeral constants are used.

        data : np.array
            An np.array of 2D np.arrays. At the top layer, the
            list is split into training and validation datasets.
            Next, into the actual data with output followed by
            each input. That is, a row of data is of the form
            y, x0, x1, ...
        """

        if hasattr(self, 'c'):

            old_error = self.fitness[0]
            old_c = self.c
            old_f = self.f

            self.evaluate_fitness(data, attempts=1)

            if old_error < self.fitness[0]:

                self.fitness[0] = old_error
                self.c = old_c
                self.f = old_f

        else:

            self.evaluate_fitness(data)


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

        x_data = data[0][:, 1:].T
        y_data = data[0][:, 0]

        if compute_val_error:
            x_data_val = data[1][:, 1:].T
            y_data_val = data[1][:, 0]

        if self.params['IA']:

            # simplify to avoid false intervals
            ind = copy.deepcopy(self)

            ind.simplify()

            ind.place_exponents()

            f_string = ind.convert_lisp_to_standard_for_function_creation()

            ind2 = copy.deepcopy(ind)

            ind2.replace(old='p/', new='/')
            ind2.replace(old='%', new='/')

            f_string_for_intervals = ind2.convert_lisp_to_standard_for_interval_arithmetic()

        else:

            f_string = self.convert_lisp_to_standard_for_function_creation()

        if '#c' in self.T:

            are_consts = 'c[' in f_string
            self.f = get_function(f_string, const=are_consts)

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

                if compute_val_error:
                    self.validation_fitness = error(self.c, x_data_val, y_data_val)

            else:

                error = lambda x, y, f=self.f: np.sqrt(np.mean(np.power(f(x) - y, 2)))

                self.fitness[0] = error(x_data, y_data)

                if compute_val_error:
                    self.validation_fitness = error(x_data_val, y_data_val)

        else:

            self.f = get_function(f_string)

            error = lambda x, y, f=self.f: np.sqrt(np.mean(np.power(f(x) - y, 2)))

            self.fitness[0] = error(x_data, y_data)

            if compute_val_error:
                self.validation_fitness = error(x_data_val, y_data_val)

        if self.params['IA']:

            are_consts = 'c[' in f_string_for_intervals
            f = get_function(f_string_for_intervals, const=are_consts)

            # use IA to check if solution is undefined on data interval
            try:
                if are_consts:

                    output = f(self.c, self.params['interval'])

                else:

                    output = f(self.params['interval'])
            except ZeroDivisionError:
                print('ZeroDivisionError', f_string_for_intervals)
                exit()

            # if true, set error to inf and age to inf (essentially delete individual)
            if inf in output or -inf in output:

                self.fitness[0] = float('inf')
                self.fitness[1] = float('inf')


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

        try:

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(self.c, x_data) - y_data, 2)))

        except (TypeError, AttributeError) as e:   # if no var self.c

            self.testing_fitness = np.sqrt(np.mean(np.power(self.f(x_data) - y_data, 2)))

        return self.testing_fitness


    def get_effort_tree_eval(self, datasets):
        """Each non-leaf node is an operation, so
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

        num_leaves, num_nodes = self.get_num_leaves()
        num_nonleaves = num_nodes - num_leaves

        effort_per_eval = num_nonleaves

        num_data_points = len(datasets[0]) + len(datasets[1])

        num_ops_per_RMSE = 3*num_data_points + 1

        return num_ops_per_RMSE + num_data_points*effort_per_eval


    def __eq__(self, other):
        """In terms of the pareto front. Two individuals are
        considered identical if their fitness (in all objectives)
        are equal. I have arbiarily chosen to round to 7 decimal places.

        Parameters
        ----------
        other : Individual
            The individual to be compared with self.

        Examples
        --------
        >>> I1 = I2 = GP.Individual(np.random.RandomState(0),
                            primitive_set=['*', '+', '-'],
                            terminal_set=['#x'])
        >>> I1.fitness = np.array([1., 5.])
        >>> I2.fitness = np.array([1., 5.])
        >>> I1 == I2
        True
        """

        return round(self.fitness[0], 7) == round(other.fitness[0], 7) and int(self.fitness[1]) == int(other.fitness[1])
