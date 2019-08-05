from .common_functions import union, get_function
from .Tree import Tree
from .Individual import Individual
from .consts import *
from .protected_functions import *

import numpy as np


class IndividualRestricted(Individual):
    """This class is a individual in a symbolic
    regression algorithm. It can be given restrictions
    to the structure of the GP tree."""

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
        params : dict
            'restrictions' : dict
                The tree restrictions. key=location, value=label.
                If a node must be a primitive due to the locations
                of the restrictions, the value will be '_p_'
        """

        self.restrictions = params['restrictions']

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

                    self.tree = self.generate_individual_full_recursive(depth)

                else:

                    self.tree = self.generate_individual_grow_recursive(depth)

            elif method == 'grow':

                self.tree = self.generate_individual_grow_recursive(depth)

            elif method == 'full':

                self.tree = self.generate_individual_full_recursive(depth)

            else:

                print("Error:", method, "is an unknown method.")

            self.apply_rules_to_tree()


    # ---------------------------------------------------------------------- #
    #                       Tree Creation Functions
    # ---------------------------------------------------------------------- #

    def generate_individual_full_recursive(self, max_depth, location=(), tree=None, depth=0):
        """Generate a tree using the restricted full method.
        This method creates each node from the primitive set
        until max depth is reached at which point the node is
        selected from the terminal set. However, if a restriction
        is encountered, a specific node will be placed. Also,
        max_depth may be ignored if the restrictions and max_depth
        conflict.

        Parameters
        ----------
        max_depth : int
            This parameter determine the max depth of
            the tree returned by this function. The tree
            generated by this method will have depth of
            exactly max_depth. This number will be ignored
            if it conflicts with self.restrictions
        location : tuple (defaul=())
            The location of the current node. This is the
            child indices required to get to this node
            from the root.
        tree : list of lists (default=None)
            The tree as list of lists. The default is actually
            [].
        depth : int (default=0)
            The depth that the tree is currently at.

        Examples
        --------
        >>> I = GP.IndividualRestricted(rng=np.random.RandomState(0),
                                        primitive_set=['+', '*', '-', '%'],
                                        terminal_set=['#x', '#c'],
                                        restrictions={(): '%',
                                                      (0, 0): '#x'})

        >>> I.tree = I.generate_individual_full_recursive(2)
        >>> I.get_lisp_string()
        (% (% (#x) (#c)) (% (#c) (#c)))
        """
        if tree is None:

            tree = []

            # get min_depth based on self.restrictions
            max_key, max_value = max(self.restrictions.items(), key=lambda x: len(x[0]))

            min_depth = len(max_key)

            if max_value in self.P:
                min_depth += 1

            # The location is () by default in which case
            # len(location) has no effect in the following comparison.
            # But, if a location is specified (not the root location)
            # then max_depth only refers to the depth of the subtree
            # so len(location) is added.
            assert min_depth <= len(location) + max_depth, ('The max_depth is not large enough for self.restrictions. '
                                                            'The deepest node in self.restrictions is '+str(max_value)+' at location '+str(max_key)+''
                                                            ' so max depth should be at least '+str(min_depth)+'.')

        if location in self.restrictions:

            tree.append(self.restrictions[location])

            if self.restrictions[location] in required_children:

                for i in range(required_children[self.restrictions[location]]):
                    tree.append(self.generate_individual_full_recursive(depth=depth+1,
                                                                        location=(*location, i),
                                                                        max_depth=max_depth))

        elif depth == max_depth:
            tree.append(self.rng.choice(self.T))

        elif depth < max_depth:

            primitive = self.rng.choice(self.P)

            tree.append(primitive)

            for i in range(required_children[primitive]):
                tree.append(self.generate_individual_full_recursive(depth=depth+1,
                                                                    location=(*location, i),
                                                                    max_depth=max_depth))

        return tree


    def generate_individual_grow_recursive(self, max_depth, location=(), tree=None, depth=0):
        """Generate a tree using the restricted grow method.
        This method creates each node and determines how to proceed
        based on the label selected. This continues until max depth
        is reached at which point the node is selected from the
        terminal set. However, if a restriction
        is encountered, a specific node will be placed. Also,
        max_depth may be ignored if the restrictions and max_depth
        conflict. The tree is guaranteed to be at least as big as
        self.restrictions requires.

        Parameters
        ----------
        max_depth : int
            This parameter determine the max depth of
            the tree returned by this function. The tree
            generated by this method will have depth of
            exactly max_depth. This number will be ignored
            if it conflicts with self.restrictions
        location : tuple (defaul=())
            The location of the current node. This is the
            child indices required to get to this node
            from the root.
        tree : list of lists (default=None)
            The tree as list of lists. The default is actually
            [].
        depth : int (default=0)
            The depth that the tree is currently at.

        Examples
        --------
        >>> I = GP.IndividualRestricted(rng=np.random.RandomState(0),
                                        primitive_set=['+', '*', '-', '%'],
                                        terminal_set=['#x', '#c'],
                                        restrictions={(): '%',
                                                      (0, 0): '#x'})

        >>> I.tree = I.generate_individual_grow_recursive(2)
        >>> I.get_lisp_string()
        (% (% (#x) (#c)) (% (#c) (#c)))
        """

        if tree is None:

            tree = []

            # get min_depth based on self.restrictions
            max_key, max_value = max(self.restrictions.items(), key=lambda x: len(x[0]))

            min_depth = len(max_key)

            if max_value in self.P:
                min_depth += 1

            # The location is () by default in which case
            # len(location) has no effect in the following comparison.
            # But, if a location is specified (not the root location)
            # then max_depth only refers to the depth of the subtree
            # so len(location) is added.
            assert min_depth <= len(location) + max_depth, ('The max_depth is not large enough for self.restrictions. '
                                                            'The deepest node in self.restrictions is '+str(max_value)+' at location '+str(max_key)+''
                                                            ' so max depth should be at least '+str(min_depth)+'.')

        if location in self.restrictions:

            tree.append(self.restrictions[location])

            if self.restrictions[location] in required_children:

                for i in range(required_children[self.restrictions[location]]):
                    tree.append(self.generate_individual_grow_recursive(depth=depth+1,
                                                                        location=(*location, i),
                                                                        max_depth=max_depth))

        elif depth == max_depth:
            tree.append(self.rng.choice(self.T))

        elif depth < max_depth:

            location_prefix = [key[:len(location)] for key in self.restrictions if len(key) >= len(location)]

            if np.any([location == prefix for prefix in location_prefix]):
                label = self.rng.choice(self.P)

            else:
                label = self.rng.choice(self.C)

            tree.append(label)

            if label in required_children:

                for i in range(required_children[label]):
                    tree.append(self.generate_individual_grow_recursive(depth=depth+1,
                                                                        location=(*location, i),
                                                                        max_depth=max_depth))

        return tree

    # ----------------------------------------------------------------------------- #
    #                                  Mutations
    # ----------------------------------------------------------------------------- #

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

        new_subtree = self.generate_individual_grow_recursive(min(mutation_param, self.max_depth - depth), location=child_indices)

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

        if child_indices in self.restrictions:

            leaf[0] = self.restrictions[child_indices]

        else:

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

        if child_indices in self.restrictions:

            leaf[0] = self.restrictions[child_indices]

        else:

            if '#x' in self.T:
                leaf[0] = 'x'+str(self.rng.choice(self.num_vars))

        return subtree
