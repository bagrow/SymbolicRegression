from .consts import *

import pygraphviz as pgv

import collections
import copy


class Tree:
    """This class represents a tree as a list of lists.
    Each sublist specifies the child nodes of the current node."""

    def __init__(self, tree, num_vars=1, rng=None, **params):
        """Initialize tree

        Parameters
        ----------
        tree : list (of lists) or str
            If list, this will become the tree. If str, the function
            Tree.from_string will be called to convert the assumed
            lisp to a list of lists.
        num_vars : int (default=1)
            The number of input variables that the tree accepts. All
            input variables are of the form x0, x1, ...
        rng : Random number generator (optional)
            This allows initial seed to carry through layers of class
            and get reproducible runs. Example: rng=np.random.RandomState(0).
            If not specified, running certain methods will not be possible.
        params : key word arguments
            Stuff for child classes.

        Examples
        --------
        >>> t = GP.Tree(tree=['*', ['#x'], ['#f']], rng=np.random.RandomState(0))
        >>> t.get_lisp_string()
        '(* (x0) (4.881350392732472))'

        >>> t = GP.Tree(tree='(% (1) (x0))')
        >>> t.get_lisp_string()
        '(% (1) (x0))'
        """

        self.rng = rng
        self.num_vars = num_vars
        self.params = params

        if type(tree) == list:
            self.tree = tree

        elif type(tree) == str:
            self.from_string(tree)

        elif tree is None:
            self.tree = tree

        else:
            print('ERROR in Tree.__init__: Unkonwn type for arugment tree')
            print('Type is ', type(tree))
            exit()

        self.apply_rules_to_tree()


    def apply_rules_to_tree(self, tree=None):
        """Check each node name in tree (self) to see
        if the name needs adjustment. This could be a
        constant is specified but not given a value
        or a variable is specified but not which variable.

        Parameters
        ----------
        tree : list (of lists, default=self.tree)
            A subtree of self.tree may be specified.
            Tree contains all the nodes to which the
            rules will be applied.

        Examples
        --------
        >>> t = GP.Tree(tree=['*', ['#x'], ['#f']], rng=np.random.RandomState(0))
        >>> t.get_lisp_string()
        '(* (#x) (#f))'
        >>> t.apply_rules_to_tree()
        >>> t.get_lisp_string()
        '(* (x0) (4.881350392732472))'
        """

        tree = self.tree if tree is None else tree

        if type(tree) == list:

            tree[0] = self.apply_rules_to_node(node=tree[0])

            for t in tree[1:]:

                self.apply_rules_to_tree(tree=t)

        else:

            tree = self.apply_rules_to_node(node=tree)


    def apply_rules_to_node(self, node):
        """Here is the actual list of rules. The
        rules are technically only applied to the
        nodes. In the previous function the rules
        are applied to each node (thus applied to
        the tree).

        Rules
        #i becomes an integer
        #f becomes a float
        #c becomes a constant named c0, c1, ..., c9
        #x becomes a variable named x0, x1, ..., x9

        Parameters
        ----------
            node : str
        """

        if node == "#i":

            return self.rng.randint(-50, 50)

        elif node == "#f":

            return self.rng.uniform(-50, 50)

        elif node == "#c":

            return 'c' + str(self.rng.choice(10))

        elif node == "#x":

            return 'x' + str(self.rng.choice(self.num_vars))

        else:

            return node

# ------------------------------------------------------------ #
#                   Get Tree as String
# ------------------------------------------------------------ #

    def get_lisp_string(self, subtree=None, actual_lisp=False):
        """Get string (lisp) representing the tree. Since every
        element of self.tree is a list, it is easiest to write
        the lisp with terminals surround by parenthesis like:
        (* (3) (1)) instead of (* 3 1)

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            Can use this argument to get lisp of a
            subtree rather than the entire tree.
        actual_lisp : bool (default=False)
            If True, get the actual lisp. That means
            no parenthesis around terminal nodes.

        Returns
        -------
        lisp : str
            The lisp string with the mentioned twist.

        Examples
        --------
        >>> tree = GP.Tree('(* (3) (1))')
        >>> tree.get_lisp_string()
        >>> tree.get_lisp_string(actual_lisp=True)
        (* (3) (1))
        (* 3 1)
        """

        if subtree is None:
            lisp = str(self.tree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')

        else:
            lisp = str(subtree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')

        if actual_lisp:

            fixed_lisp_list = []

            for word in lisp.split(' '):

                if '(' in word and ')' in word:
                    fixed_lisp_list.append(word[1:-1])

                else:
                    fixed_lisp_list.append(word)

            lisp = ' '.join(fixed_lisp_list)

        return lisp


    def convert_lisp_to_standard(self, convertion_dict):
        """General versions of this function where
        conversion is specified by a dictionary.
        This function also forces input variables
        to be involved in the expression so that
        vectorization can be achieved (using eval).

        Examples
        --------
        >>> tree = GP.Tree('(* (3) (1))')
        >>> tree.get_lisp_string()
        (* (3) (1))
        >>> tree.convert_lisp_to_standard(convertion_dict={'*': 'Apple'})
        Apple(3,1)+0*x[0]
        """

        lisp = str(self.tree).replace(',', '').replace('\'', '')

        stack = ['']
        standard = ""

        split_lisp = lisp.split()

        # Check if single node function to avoid added a comma at the end of expr
        if len(split_lisp) == 1:

            # var = 'x' if self.num_vars == 1 else 'x[0]'
            var = 'x[0]'

            if lisp[1] == 'c' and lisp[2].isdigit():

                # the 0*x avoids vectorization problems.
                return 'c[' + lisp[2] + ']+0*' + var

            elif lisp == 'x':

                return lisp

            elif lisp[1] == 'x' and lisp[2].isdigit() and len(lisp) == 4:

                return 'x[' + lisp[2] + ']'

            else:

                return lisp + '+0*' + var

        for word in split_lisp:

            if word[0] == '[' and word[-1] == ']':

                count = word.count(']')

                if word[1] == 'c' and word[-count - 1].isdigit():

                    standard += 'c[' + word[-count - 1] + ']'

                elif word[1] == 'x' and word[-count - 1].isdigit() and len(word) == 3 + count:

                    standard += 'x[' + word[-count - 1] + ']'

                else:

                    standard += word[1:-count]

                count -= 1

                for _ in range(count):
                    stack.pop()

                standard += ")" * count

                if len(stack) > 1:
                    standard += ','

            elif word[0] == '[':

                if word[1:] in convertion_dict and word[1:] in required_children_for_function:

                    stack.append(convertion_dict[word[1:]])
                    standard += stack[-1] + '('

                else:

                    print('ERROR in convert_lisp_to_standard: '
                          'bad function ', word[1:])
                    exit()

            elif word[-1] == ']':

                count = word.count(']')

                standard += word[:-count]

                for _ in range(count):
                    stack.pop()

                standard += ')' * count

                if len(stack) > 1:
                    standard += ','

            else:

                standard = standard + word + ','

        if 'x' not in standard:  # assumes only x is for the variable

            # var = 'x' if self.num_vars == 1 else 'x[0]'
            var = 'x[0]'

            standard += '+0*' + var  # to avoid vectorization issue

        return standard


    def convert_lisp_to_standard_for_function_creation(self):
        """Convert the string version of self.tree to a typical
        layout for an equation. This method uses math_translate
        dictionary (from consts.py). This writes the string
        with the correct function names, which makes it
        interpretable by the computer.

        Examples
        --------
        >>> tree = GP.Tree('(* (3) (1))')
        >>> tree.get_lisp_string()
        (* (3) (1))
        >>> tree.convert_lisp_to_standard_for_function_creation()
        np.multiply(3,1)+0*x[0]
        """

        standard = self.convert_lisp_to_standard(convertion_dict=math_translate)

        return standard


    # def convert_lisp_to_standard_for_interval_arithmetic(self, lisp_str=None):
    #     """Convert the string to a typical layout for an equation. The string is expected
    #     to be a lisp expression. get_lisp_string() will be called if no string is passed.

    #     If a function that calls numpy is in the tree, numpy will be used. For basic operation though,
    #     standard python function will be used."""

    #     standard = convert_lisp_to_standard(convertion_dict=interval_arithmetic)

    #     return standard

# ------------------------------------------------------------ #
#                      Build Trees
# ------------------------------------------------------------ #

    def from_string(self, expression, actual_lisp=False):
        """Construct a tree from a lisp string.
        The lisp should have parenthesis around
        terminal nodes to make this conversion
        easier. The tree is saved to self.tree
        as a list of lists.

        Parameters
        ----------
        expression : str
            The lisp string

        Examples
        --------
        The first two lines of this example can be performed
        with tree = GP.Tree('(+ (x0) (1))')

        >>> tree = GP.Tree(None)
        >>> tree.from_string('(+ (x0) (1))')
        >>> tree.get_lisp_string()
        (+ (x0) (1))

        >>> tree = GP.Tree(None)
        >>> tree.from_string('(+ x0 (* c1 3))', actual_lisp=True)
        >>> tree.get_lisp_string()
        >>> tree.get_lisp_string(actual_lisp=True)
        (+ (x0) (* (c1) (3)))
        (+ x0 (* c1 3))
        """

        sbracket_expression = expression.replace('(', '[').replace(')', ']').replace(' ', ', ')

        for primitive in math_translate.keys():

            if primitive != '-':

                sbracket_expression = sbracket_expression.replace(primitive+',', '\'' + primitive + '\',')

            else:

                # check if - is a negative sign
                minus_index = [i for i, c in enumerate(sbracket_expression) if c == '-' and sbracket_expression[i+1] == ',']

                sbracket_expression = ''.join(['\'-\'' if i in minus_index else c for i, c in enumerate(sbracket_expression)])

        if actual_lisp:
            prefix = '[\''
            suffix = '\']'

            # find constants that give actual number
            string_list = []

            for word in sbracket_expression.split(' '):

                try:
                    float(word.replace('[', '').replace(']', ''))

                except ValueError:
                    string_list.append(word)

                else:
                    string_list.append('[' + word + ']')


            sbracket_expression = ' '.join(string_list)

        else:
            prefix = suffix = '\''

        for var in range(self.num_vars):
            sbracket_expression = sbracket_expression.replace('x'+str(var), prefix + 'x'+str(var) + suffix)

        for var in range(10):
            sbracket_expression = sbracket_expression.replace('c'+str(var), prefix + 'c'+str(var) + suffix)

        self.tree = eval(sbracket_expression)

# ------------------------------------------------------------ #
#                   Get Tree Info
# ------------------------------------------------------------ #

    def is_leaf(self, subtree=None):
        """Check if the current node is a leaf (has no children).

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            The subtree to which may or may not be a leaf node.

        Returns
        -------
        True if subtree is a leaf node otherwise False.
        """

        subtree = self.tree if subtree is None else subtree

        if len(subtree) == 1:

            return True

        else:

            return False


    def is_constant(self, subtree):
        """Check if the current node
        is a constant. If the tree
        has been constructed correctly,
        this would also indicate that
        the node is a leaf. (However,
        not all leaves are constant.)

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            The subtree to which may or may not be a constant node.

        Returns
        -------
        True if subtree is a constant node otherwise False.
        """

        if type(subtree[0]) == str or type(subtree[0]) == np.str_:

            if subtree[0][0] == 'c' and subtree[0][1].isdigit():

                return True

            else:

                return False

        else:

            return True


    def is_variable(self, subtree):
        """Check if current node is a variable.

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            The subtree to which may or may not be a variable node.

        Returns
        -------
        True if subtree is a variable node otherwise False.
        """

        if type(subtree[0]) == str or type(subtree[0]) == np.str_:

            if subtree[0] == 'x':

                return True

            elif len(subtree[0]) == 2:

                if subtree[0][0] == 'x' and subtree[0][1].isdigit():

                    return True

        return False


    def select_subtree(self, child_indices, subtree=None):
        """Find the node associated with the index list and return it.

        Parameters:
        ----------
        child_indices : iterable
            The indices of children. For example, the child_indices
            of the 0-th child of the root node would be [0].
        subtree : list (of lists, default=self.tree)
            current subtree

        Examples
        --------
        >>> tree = GP.Tree('(* (x0) (* (x1) (x2)))', num_vars=3)
        >>> tree.select_subtree(child_indices=(1, 0))
        ['x1']
        """

        subtree = self.tree if subtree is None else subtree

        if child_indices == [] or child_indices == ():

            return subtree

        else:

            index = child_indices[0] if type(subtree[0]) == list else child_indices[0]+1

            return self.select_subtree(subtree=subtree[index], child_indices=child_indices[1:])


    def set_subtree(self, new_subtree, child_indices, subtree=None):
        """Find and set the node referenced by child_indices equal
        to new_node.

        Parameters
        ----------
        new_subtree : list (like self.tree)
            The subtree that will be placed in the desired location.
        child_indices : iterable (of ints)
            The location of the node that where new_subtree should
            be placed. Iterable is the indices of children.
            For example, the child_indices of the 0-th child of root
            node would be [0].  Root node is indicated by [] or ().
        subtree : list (default=self.tree)
            Current subtree the child_index_list is based on.

        Examples
        --------
        >>> tree = GP.Tree('(* (x0) (* (x1) (x2)))', num_vars=3)
        >>> tree.set_subtree(new_subtree=['+', [1], [2]],
                             child_indices=(1, 0))
        >>> tree.get_lisp_string()
        (* (x0) (* (+ (1) (2)) (x2)))
        """

        subtree = self.tree if subtree is None else subtree

        if child_indices == [] or child_indices == ():

            # [:] means the locations of stored values of subtree
            # will be changed.
            # Thus, self.tree will be effected by this assignment.
            subtree[:] = new_subtree

        else:

            index = child_indices[0] if type(subtree[0]) == list else child_indices[0] + 1

            return self.set_subtree(new_subtree=new_subtree,
                                    subtree=subtree[index],
                                    child_indices=child_indices[1:])


    def get_node_list(self, prefix=(), node_list=None, subtree=None):
        """Get a list of all nodes below subtree (including itself).
        Each node is represented by an iterable of child indices.
        For example, [0, 1, 1] refers to the 0-th child's 1-st child's
        1-st child. The parameters are mostly for recursion,
        and in most instances need not be specified.

        Parameters
        ----------
        prefix : tuple (default=())
            For recursion. This argument keeps track of the
            child indices. It is named prefix because it is
            appended to with the new child nodes.
        node_list : list (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_list : list (of tuples)
            The locations of all nodes in the tree as child
            indices. Note that the root node is empty tuple.

        Examples
        --------
        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_list()
        [(), (0,), (1,), (1, 0), (1, 1)]
        """

        subtree = self.tree if subtree is None else subtree

        if node_list is None:
            node_list = [()]

        # if subtree is a single node
        if len(subtree) == 1:
            return node_list

        else:
            for i, st in enumerate(subtree):

                if type(st) == list:
                    node_list.append((*prefix, i-1))
                    node_list.extend(self.get_node_list(prefix=(*prefix, i-1), node_list=[], subtree=st))

            return node_list


    def get_node_dict(self, prefix=(), node_dict=None, subtree=None):
        """Get a dictionary of all nodes in subtree. Each node is
        represented by a label and a list of child indices.
        The list [0, 1, 1] refers to the 0-th child's 1-st
        child's 1-st child. The parameters are mostly for
        recursion, and in most instances need not be
        specified.

        Parameters
        ----------
        prefix : tuple (default=())
            For recursion. This argument keeps track of the
            child indices. It is named prefix because it is
            appended to with the new child nodes.
        node_dict : dict (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_dict : list (of tuples)
            The locations of all nodesin the tree as child
            indices (these are the keys) and the node labels
            (these the values).

        Examples
        --------
        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_dict()
        {(1, 0): 'x0', (): '-', (1,): '+', (0,): 3, (1, 1): 3}
        """

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:
            node_dict = {(): subtree[0]}

        else:
            node_dict[prefix] = subtree[0]

        if len(subtree) == 1:
            return node_dict

        else:

            for i, st in enumerate(subtree):

                if type(st) == list:
                    self.get_node_dict(prefix=(*prefix, i - 1), subtree=st, node_dict=node_dict)

            return node_dict


    def get_node_map(self, loc=(), node_dict=None, subtree=None):
        """Look through all nodes and record nodes in a dictionary.
        This dictionary will fill all location of node names in the tree.

        Parameters
        ----------
        loc : tuple (default=())
            For recursion. This argument keeps track of the
            locations.
        node_dict : dict (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_dict : list (of tuples)
            The node labels (these the keys). The locations
            of all nodes with the same label are stored in
            a set (these are the keys).

        Examples
        --------
        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_map()
        {'-': {()}, 3: {(0,), (1, 1)}, 'x0': {(1, 0)}, '+': {(1,)}}
        """

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:
            node_dict = {}

        if subtree[0] in node_dict:
            node_dict[subtree[0]] = node_dict[subtree[0]].union({loc})

        else:
            node_dict[subtree[0]] = {loc}  # set literal

        if len(subtree) == 1:
            return node_dict

        else:
            for i, st in enumerate(subtree):
                if type(st) == list:
                    self.get_node_map(loc=(*loc, i - 1), subtree=st, node_dict=node_dict)


            return node_dict


    def get_tree_size(self, subtree=None, tree_size=0, special_counts=None):
        """Count the number of nodes in the tree and return it.

        Parameters
        ----------
        subtree : list (of lists, defaul=self.tree)
            The subtree from which nodes are counted.subtree
        tree_size : int (default=0)
            For recursion. Keeps track of number of nodes.
        special_counts : dict (optional)
            Use this to count certain nodes (keys) as more
            any number you like (values).

        Returns
        -------
        tree_size : int (or float)
            The number of nodes in the tree. This could be
            a float if some values of special_counts are
            not integers and the corresponding label is in
            the tree.

        Examples
        --------
        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_tree_size()
        5

        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_tree_size(special_counts={'+': 2})
        6

        """

        if subtree is None:
            subtree = self.tree

        if special_counts is None or subtree[0] not in special_counts:
            tree_size += 1

        else:
            tree_size += special_counts[subtree[0]]

        if len(subtree) == 1:
            return tree_size

        else:

            for st in subtree:

                if type(st) == list:
                    tree_size = self.get_tree_size(subtree=st,
                                                   tree_size=tree_size,
                                                   special_counts=special_counts)

            return tree_size


    def get_depth(self):
        """Find the deepest node in the tree and
        return the depth. This is done
        by looking at the list of all nodes.

        Returns
        -------
        max_depth : int
            The largest number of links between any
            node and the root node.

        Examples
        --------
        >>> tree = GP.Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_depth()
        2
        """

        node_list = self.get_node_list()

        if node_list == [()]:
            return 0

        node_depth = [len(node_str) for node_str in node_list]

        return max(node_depth)


    def draw_tree(self, save_loc):
        """Draw tree using pygraphviz. Eventually,
        would like to make it possible to use LaTeX
        for the labels. There is a Python package
        (dot2tex).

        Parameters
        ----------
        save_loc : str
            The image generated will be saved
            in the location specified."""

        vis_tree = pgv.AGraph()

        node_dict = self.get_node_dict()

        # need to make sure that nodes are ordered for - or /
        m = max(map(len, node_dict.keys()))
        sort_key = lambda x: 2**(len(x[0])+m) + sum([2**xi for xi in x[0]])

        node_dict = collections.OrderedDict(sorted(node_dict.items(), key=sort_key))

        edgeColor = 'black'
        nodeColor = 'whitesmoke'

        for key in node_dict:

            if key == '':
                fake_key = 'root'

            else:
                fake_key = key


            try:
                float(node_dict[key])

            except ValueError:
                label = node_dict[key]

            else:
                label = '%.2E' % node_dict[key]


            vis_tree.add_node(fake_key, label=label,
                              fixedsize=False,
                              style='filled',
                              color=edgeColor,
                              shape='circle',
                              fillcolor=nodeColor)

            if key is not '':

                if len(key) == 1:
                    parent = 'root'

                else:
                    parent = key[:-1]

                vis_tree.add_edge(parent, key)

        vis_tree.draw(save_loc, prog='dot')

# ------------------------------------------------------------ #
#                   Simplification
# ------------------------------------------------------------ #

    def simplify(self):
        """Use some simple simplification rules to simplify the
         tree with root node equal to self. To use this method,
         simplification rules must be supplied. They are stored
         in consts.py"""

        count = self.simplify_once()

        if count != 0:
            self.simplify()


    # def simplify_consts(self):
    #     """The specific simplification function only simplifies constants. It combines
    #     subtrees that only contain constant nodes. EVENTUALLY WANT THIS TO WORK
    #     FOR EPHEMERAL CONSTANTS. CURRENTLY ONLY WORKS IF CONSTANTS ARE COMPUTED.

    #     :return: True if simplification was applied. False otherwise.
    #     """
    #     node_dict = self.get_node_dict()

    #     # get location of all none
    #     x_locs = [key for key in node_dict if 'x' in node_dict[key]]

    #     # remove parents of x_locs from node_dict
    #     to_remove = set()

    #     for x_key in x_locs:

    #         for key in node_dict:

    #             if is_elder(child=x_key, elder=key):
    #                 to_remove.add(key)

    #     new_node_dict = {}

    #     for key in to_remove:
    #         new_node_dict[key] = node_dict[key]
    #         del node_dict[key]

    #     if len(node_dict) == 0:
    #         return self

    #     # group into branches
    #     branches = get_branches(node_dict)

    #     # count the constants in each branch
    #     counts = []

    #     for branch in branches:
    #         counts.append(count_consts(branch))

    #     # find branches that have the same constants involved
    #     same_const_branches = []

    #     for i, count in enumerate(counts):

    #         for j, count2 in enumerate(counts[i + 1:]):

    #             if count.keys() == count2.keys():
    #                 same_const_branches.append((i, i + 1 + j))

    #     # use the counts to simplify the branches
    #     const_num = 0

    #     for i, j in same_const_branches:
    #         new_node_dict[first_key(branches[i])] = 'c' + str(const_num)
    #         new_node_dict[first_key(branches[j])] = 'c' + str(const_num)
    #         const_num = 0 if const_num == 9 else 1 + const_num

    #     # put from largest to smallest and delete these branches
    #     reordered_same_const_branches = \
    #         sorted(np.unique(np.array(same_const_branches).flatten()), reverse=True)

    #     for i in reordered_same_const_branches:
    #         del branches[i]

    #     for branch in branches:
    #         new_node_dict[first_key(branch)] = 'c' + str(const_num)
    #         const_num = 0 if const_num == 9 else 1 + const_num

    #     # create node from new_node_dict
    #     return tree_from_dict(Node=self.__class__,
#                               dict=new_node_dict,
#                               rng=self.rng,
#                               **self.params)


    def handle_simplification_rules1(self, rule, count, subtree):
        """Handle rules where the parent node has only one
        child node.

        Parameters
        ----------
        rule : tuple
            The first element is the child label
            necessary to make the rule applicable.
            The second element is what this subtree
            simplifies to if the child node is in
            fact equal to the first element of this
            tuple.
        count : int
            This agrgument is used to keep track
            of how many simplification has been
            performed. This is used by self.simplify
            to determine when to stop trying to simplify.
        subtree : list (of lists)
            This is the subtree.

        Returns
        -------
        count : int
            The cumulative number of simplifications
            performed. Since count is passed into
            this method, it is possible for a mutation
            not to be performed in this function but
            count be non-zero.
        """

        child, value = rule

        child_act = self.get_lisp_string(subtree[1])

        if child == child_act:

            subtree[:] = value
            count += 1

        return count


    def handle_simplification_rules2(self, rule, count, subtree):
        """Handle the simplification rules where the parent node
        has two child nodes.

        Parameters
        ----------
        rule : tuple
            The first element is the left child label
            necessary to make the rule applicable.
            The second element is the right child label
            necessary to make the rule applicable.
            The third element is what this subtree
            simplifies to if the child nodes are in
            fact equal to the first and second element
            of this tuple. The string '&' indicates
            that the child node can be anything.
        count : int
            This agrgument is used to keep track
            of how many simplification has been
            performed. This is used by self.simplify
            to determine when to stop trying to simplify.
        subtree : list (of lists)
            This is the subtree.

        Returns
        -------
        count : int
            The cumulative number of simplifications
            performed. Since count is passed into
            this method, it is possible for a mutation
            not to be performed in this function but
            count be non-zero.
        """

        lchild, rchild, value = rule

        lchild_act = self.get_lisp_string(subtree[1])
        rchild_act = self.get_lisp_string(subtree[2])

        if lchild == '&' == rchild:

            if lchild_act == rchild_act:

                subtree[:] = value
                count += 1

        elif lchild == lchild_act and rchild == rchild_act:

            subtree[:] = value
            count += 1

        elif lchild == '&' and rchild == rchild_act:

            if value != '&':

                subtree[:] = value

            else:

                subtree[:] = copy.deepcopy(subtree[1])

            count += 1

        elif lchild == lchild_act and rchild == '&':

            if value != '&':

                subtree[:] = value

            else:

                subtree[:] = copy.deepcopy(subtree[2])

            count += 1

        return count


    def simplify_once(self, subtree=None):
        """Apply the simple simplification
        rules involving only x, 0, and 1.

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            The subtree to simplify

        Returns
        -------
        count : int
            The total number of simplifications
            performed.
        """

        subtree = self.tree if subtree is None else subtree

        count = 0

        if len(subtree) == 1:

            return count

        else:

            if subtree[0] in simplification_rules:

                for rule in simplification_rules[subtree[0]]:

                    if required_children_for_function[subtree[0]] == 2:

                        precount = copy.copy(count)

                        count = self.handle_simplification_rules2(rule, count, subtree)

                        if count - precount > 0:
                            break

                    elif required_children_for_function[subtree[0]] == 1:

                        precount = copy.copy(count)

                        count = self.handle_simplification_rules1(rule, count)

                        if count - precount > 0:
                            break

                    else:

                        print('simplification method not implemented for functions with',
                              required_children_for_function[subtree[0]], 'children')

            for child in subtree[1:]:

                count += self.simplify_once(subtree=child)

        return count

# ------------------------------------------------------------ #
#                   Tree Manipulation
# ------------------------------------------------------------ #

    def place_exponents(self):
        """Find x's connected by multiplication and
        rewrite the tree so that x**n is used instead of
        multiplication of n x's.

        In interval arithmetic, [-1, 1]*[-1, 1] is
        different than [-1, 1]^2. This simplification
        can avoid the problem."""

        node_map = self.get_node_map()

        x_keys = [key for key in node_map if 'x' in str(key)]

        for i in range(self.num_vars):

            # var = 'x['+str(i)+']' if self.num_vars > 1 else 'x'
            var = 'x[' + str(i) + ']'

            x_locs = []
            x_names = []

            for x in x_keys:

                if var in x:
                    x_locs.extend(node_map[x])
                    x_names.extend([x] * len(node_map[x]))

            if not x_locs or len(x_locs) == 1:
                continue

            changed_tree = False

            # look at pairs of x's
            for j, (l1, n1) in enumerate(zip(x_locs[:-1], x_names[:-1])):

                for l2, n2 in zip(x_locs[j + 1:], x_names[j + 1:]):

                    # make loc2 the deeper location
                    name1, name2 = (n2, n1) if len(l2) < len(l1) else (n1, n2)
                    loc1, loc2 = (l2, l1) if len(l2) < len(l1) else (l1, l2)

                    elder = find_youngest_common_elder(loc1, loc2)

                    if is_consistent_operation_between(elder, loc1, '*', node_map) and \
                            is_consistent_operation_between(elder, loc2, '*', node_map):
                        node = self.select_subtree(child_index_list=loc1.split(','))
                        exp1 = 1 if len(name1) <= 2 else int(name1[-1])
                        exp2 = 1 if len(name2) <= 2 else int(name2[-1])
                        node.name = var + '**' + str(exp1 + exp2)

                        other_child = get_other_child(loc2)

                        parent = get_parent(other_child)

                        n_parent = self.select_subtree(child_index_list=parent.split(','))
                        n_other_child = self.select_subtree(child_index_list=other_child.split(','))
                        n_parent.set(n_other_child)
                        changed_tree = True

                        self.place_exponents()
                        break

                if changed_tree:
                    break

            if changed_tree:
                break


    def replace(self, old, new, subtree=None):
        """Replace all occurrences of old with new. This function
        does not check if this is an acceptable substitution.

        Parameters
        ----------
        old : str
            The node label to find and replace.
        new : str
            The label to replace the old node labels with.
        subtree : list (of lists, default=self.tree)
            The subtree in which to replace old with new.
        """

        subtree = self.tree if subtree is None else subtree

        if len(subtree) == 1:

            if subtree[0] == old:

                subtree[:] = new

        for st in subtree:

            if type(st) == list:

                self.replace(old=old, new=new, subtree=st)

# ------------------------------------------------------------ #
#                      Node Functions
# ------------------------------------------------------------ #

    @staticmethod
    def get_youngest_common_elder(loc1, loc2):
        """Find the youngest common node of loc1 and loc2.

        Parameters
        ----------
        loc1 : tuple
            Location of node 1.
        loc2 : tuple
            Location of node 2.

        Returns
        -------
        common_elder : tuple
            The location of the lowest node in the tree
            that is hyperparent to both locations.

        Examples
        --------
        >>> GP.Tree.get_youngest_common_elder((1, 0, 0), (1, 0, 1))
        (1, 0)
        """

        loc1_array = np.array(loc1)
        is_equal = loc1_array == np.array(loc2)
        end = np.where(is_equal==False)[0][0]

        return tuple(loc1_array[:end])

        return common_elder


    @staticmethod
    def is_consistent_operation_between(elder, child, op, node_map):
        """Check if operation is used at all nodes between child and elder
        including elder but not child.

        Parameters
        ----------
        elder : tuple
            location of older (higher) node
        child : tuple
            location of younder (lower) node
        op : str
            Label that we want to know is between elder and child
            (Ex. '*' for multiply)
        node_map : dict
            Same kind of dictionary that you would get from
            self.get_node_map()

        Returns
        -------
        bool
            True if op is the only primitive between elder and child.
            elder must also be the primitive.

        Example
        -------
        >>> tree = GP.Tree('(* (x0) (+ (c2) (+ (x0) (c1))))')
        >>> node_map = tree.get_node_map()
        >>> GP.Tree.is_consistent_operation_between((1,), (1, 1, 1), '+', node_map)
        True
        """

        # check if op is in tree at all
        if op not in node_map:
            return False

        # otherwise ...
        current_node = child

        while current_node != elder:

            # get parent of current_node
            current_node = current_node[:-1]

            if current_node not in node_map[op]:
                return False

        return True


    @staticmethod
    def get_other_child(loc):
        """If loc is one of two children
        then return the location of the other one.

        Example
        -------
        >>> GP.Tree.get_other_child((0, 1, 0))
        (0, 1, 1)
        """

        last_digit = 1 if loc[-1] == 0 else 0

        other_child = (*loc[:-1], last_digit)

        return other_child


    @staticmethod
    def is_elder(child, elder):
        """Returns True if child is lower on the tree (in the same branch) than elder.
        Both child and elder are tuples representing the location of the node.
        The tuple is a sequence of child indices. For example,
        (0, 0) is the 0th child of the 0th child of the root node.

        Parameters
        ----------
        child : iterable
            Location of child
        elder : iterable
            Location of elder

        Returns
        -------
        bool
            True if child location begins with elder location.
        """

        # If the child is lower in the tree than elder (as expected),
        # elder should be entirely contained in the child.
        if elder in child:

            # We also need to check that elder is the beginning of
            # child. If not, elder is definitely higher than child
            # in the tree, but they are "cousins" (or something).
            if elder == child[:len(elder)]:
                return True

            # elder is higher, but they are not in the same branch
            else:
                return False

        # elder is lower than child and/or not in the same branch
        else:
            return False

    @staticmethod
    def is_same_branch(loc1, loc2):
        """Check loc1 and loc2 are in
        the same branch of the tree."""

        return is_elder(child=loc1, elder=loc2) or is_elder(child=loc2, elder=loc1)


    @staticmethod
    def get_branches(nodes):
        """Given a dictionary of at least a portion of tree, split
        the tree into its branches. If the entire tree is present, there
        should be only one branch. Return a list of branches, even
        if only one exists.

        Parameters
        ----------
        nodes : dict
            A dictionary like one returned from Tree.get_node_dict

        Returns
        -------
        branches : list
            A list of branches. Each list is a list of nodes in a branch.

        Examples
        --------
        >>> GP.Tree.get_branches({(0,): 'x0', (1,): 2})
        [OrderedDict([((0,), 'x0')]), OrderedDict([((1,), 2)])]
        """

        ordered_nodes = list(sorted(nodes.items(), key=lambda x: len(x[0])))
        branches = [collections.OrderedDict({ordered_nodes[0][0]: ordered_nodes[0][1]})]

        for loc, name in ordered_nodes[1:]:

            for branch in branches:

                if Tree.is_elder(elder=first_key(branch), child=loc):

                    branch[loc] = name
                    break

            else:

                branches.append(collections.OrderedDict({loc: name}))

        return branches


    # def count_consts(nodes):
    #     """Count how many different constants appear in the nodes.
    #     Does not work for ephemeral constants."""

    #     counts = {}

    #     for key in nodes:

    #         if 'c' in nodes[key] and 'cos' != nodes[key]:

    #             if key in counts:

    #                 counts[nodes[key]] += 1

    #             else:

    #                 counts[nodes[key]] = 1

    #     return counts

# ------------------------------------------------------------ #
#                   Used in some static methods
# ------------------------------------------------------------ #

def first_key(ordered_dictionary):
    """Get first element of an ordered dictionary"""

    return next(iter(ordered_dictionary))


def first_value(ordered_dictionary):
    """Get first value of an ordered dictionary"""

    return ordered_dictionary[first_key(ordered_dictionary)]
