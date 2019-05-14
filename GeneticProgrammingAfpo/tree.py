from .consts import *

import pygraphviz as pgv

import collections
import copy


class Tree:

    def __init__(self, tree, num_vars=1, rng=None, **params):
        """
        Initialize tree

        Parameters:
            tree: list of lists. Each sublist specifies the child nodes of the
        current node.
            num_vars: The number of input variables that the tree accepts.
            rng: Random number generator. This allows initial seed to carry
            through layers of class and get reproducible runs.
            params: Stuff for child classes.
        """

        self.rng = rng
        self.tree = tree
        self.num_vars = num_vars
        self.params = params

        self.apply_rules_to_tree()


    def apply_rules_to_tree(self, tree=None):
        """Check each node name in tree (self) to see
        if the name needs adjustment. This could be a
        constant is specified but not given a value
        or a variable is specified but not which variable.

        Parameters:
            tree: Default is self.tree, but a subtree of
            self.tree may be specified. Tree contains all
            the nodes to which the rules will be applied."""

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

        Parameters:
            node: string"""

        if node == "#i":

            return self.rng.randint(-50, 50)

        elif node == "#f":

            return self.rng.uniform(-50, 50)

        elif node == "#c":

            return 'c' + str(self.rng.choice(10))  # could use more or less constants

        elif node == "#x":

            return 'x' + str(self.rng.choice(self.num_vars))

        else:

            return node

# ------------------------------------------------------------ #
#                   Get Tree as String
# ------------------------------------------------------------ #

    def get_lisp_string(self, subtree=None):
        """Get string (lisp) representing tree."""

        if subtree is None:

            return str(self.tree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')

        else:

            return str(subtree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')


    def convert_lisp_to_standard(self, convertion_dict):
        """General verions of this function where conversion is specified
        by a dictionary."""

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
                    standard = standard + stack[-1] + '('

                else:

                    print('ERROR in convert_lisp_to_standard_for_function_creation: '
                          'bad function ', stack[-1])
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


    def convert_lisp_to_standard_no_numpy(self):
        """Convert the string to a typical layout for an equation.
        If a function that calls numpy is in the tree, numpy will be used.
        For basic operation though, standard python function will be used."""

        lisp = str(self.tree).replace(',', '').replace('\'', '')

        stack = ['']
        standard = ""

        split_lisp = lisp.split()

        # Check if single node function to avoid adding a comma at the end of expr
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

                if stack[-1] not in ('*', '+', '/', '-'):

                    standard += ','

                else:

                    if count == 0:

                        standard += stack[-1]

                for _ in range(count):
                    stack.pop()

                standard += ")" * count

                if len(stack) > 1:
                    standard += ','

            elif word[0] == '[':

                stack.append(word[1:])

                # if stack[-1] == 'abs' or stack[-1] == 'sqrt':
                if word[1:] in required_children_for_function and word[1:] in math_translate_no_numpy:

                    if stack[-1] not in ('*', '+', '/', '-'):

                        standard += math_translate_no_numpy[stack[-1]] + '('

                    else:

                        standard += '('
                else:

                    print('ERROR in convert_lisp_to_standard_for_function_creation: '
                          'bad function ', stack[-1])
                    exit()

            elif word[-1] == ']':

                count = word.count(']')

                standard += word[:-count]

                for _ in range(count):
                    stack.pop()

                standard += ')' * count + stack[-1]

            else:

                standard += word

                if stack[-1] not in ('*', '+', '/', '-'):

                    standard += ','

                else:

                    standard += stack[-1]

        if 'x' not in standard:  # assumes only x is for the variable

            # var = 'x' if self.num_vars == 1 else 'x[0]'
            var = 'x[0]'

            standard += '+0*' + var  # to avoid vectorization issue

        return standard


    def convert_lisp_to_standard_for_function_creation(self):
        """Convert the string to a typical layout for an equation.
        This is different from convert_lisp_to_standard() because it uses math_translate
        dictionary (from consts.py). This writes the string with the correct
        function names, which makes it interpretable by the computer."""

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

                if word[1:] in math_translate and word[1:] in required_children_for_function:

                    stack.append(math_translate[word[1:]])
                    standard = standard + stack[-1] + '('

                else:

                    print('ERROR in convert_lisp_to_standard_for_function_creation: '
                          'bad function ', stack[-1])
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


    def convert_lisp_to_standard_for_interval_arithmetic(self, lisp_str=None):
        """Convert the string to a typical layout for an equation. The string is expected
        to be a lisp expression. get_lisp_string() will be called if no string is passed.

        If a function that calls numpy is in the tree, numpy will be used. For basic operation though,
        standard python function will be used."""

        lisp = str(self.tree).replace(',', '').replace('\'', '')

        stack = ['']
        standard = ""

        split_lisp = lisp.split()

        # Check if constant function to avoid added a common at the end of expr
        if len(split_lisp) == 1:

            var = 'x[0]'

            if lisp[1] == 'c' and lisp[2].isdigit():

                return 'c[' + lisp[2] + ']+0*' + var  # the 0*x avoids vectorization problems.

            elif lisp == 'x':

                return lisp

            elif lisp[1] == 'x' and lisp[2].isdigit() and len(lisp) == 4:

                return 'x[' + lisp[2] + ']'

            else:

                # lisp is a number
                return 'interval(' + lisp + ')+0*' + var

        for word in split_lisp:

            if word[0] == '[' and word[-1] == ']':

                count = word.count(']')

                if word[1] == 'c' and word[-count - 1].isdigit():

                    standard += 'c[' + word[-count - 1] + ']'

                elif word[1] == 'x' and word[-count - 1].isdigit() and len(word) == 3 + count:

                    standard += 'x[' + word[-count - 1] + ']'

                else:

                    standard += 'interval(' + word[1:-count] + ')'

                count -= 1

                for _ in range(count):
                    stack.pop()

                standard += ")" * count

                if len(stack) > 1:
                    standard += ','

            elif word[0] == '[':

                if word[1:] in math_translate_interval_arithmetic and word[1:] in required_children_for_function:

                    stack.append(math_translate_interval_arithmetic[word[1:]])
                    standard = standard + stack[-1] + "("

                    stack.append(math_translate[word[1:]])
                    standard = standard + stack[-1] + '('

                else:

                    print('ERROR in convert_lisp_to_standard_for_function_creation: bad function ', word[1:])
                    exit()

            elif word[-1] == ']':

                count = word.count("]")

                standard = standard + word[:-count]

                for _ in range(count):
                    stack.pop()

                standard = standard + ")" * count

                if len(stack) > 1:
                    standard = standard + ','

            else:

                standard = standard + word + ','

        if 'x' not in standard:  # assumes only x is for the variable

            # var = 'x' if self.num_vars == 1 else 'x[0]'
            var = 'x[0]'

            standard += '+0*' + var  # to avoid vectorization issue

        return standard

# ------------------------------------------------------------ #
#                      Build Trees
# ------------------------------------------------------------ #

    def from_string(self, expression):
        """Construct a tree from a lisp string.

        Parameters:
            expression: The lisp string."""

        sbracket_expression = expression.replace('(', '[').replace(')', ']').replace(' ', ', ')

        for primitive in math_translate.keys():

            if primitive != '-':

                sbracket_expression = sbracket_expression.replace(primitive+',', '\'' + primitive + '\',')

            else:

                # check if - is a negative sign
                minus_index = [i for i, c in enumerate(sbracket_expression) if c == '-' and sbracket_expression[i+1] == ',']

                sbracket_expression = ''.join(['\'-\'' if i in minus_index else c for i, c in enumerate(sbracket_expression)])

        for var in range(self.num_vars):

            sbracket_expression = sbracket_expression.replace('x'+str(var), '\'' + 'x'+str(var) + '\'')

        for var in range(10):

            sbracket_expression = sbracket_expression.replace('c'+str(var), '\'' + 'c'+str(var) + '\'')

        self.tree = eval(sbracket_expression)


    # def tree_from_dict(dict, Node, rng=np.random.RandomState(0), node=None, loc='', **params):
#     """Node is the class of node to be created."""

#     if node is None:

#         parent = dict['']
#         loc = ''

#         n = Node(name=parent, rng=rng, **params)

#         node_from_dict(node=n, Node=Node, rng=rng, dict=dict, loc=loc, **params)

#         return n

#     # way of checking if function node
#     if node.name in required_children_for_function:

#         for i in range(required_children_for_function[node.name]):

#             child = Node(dict[loc + str(i)], rng=rng, **params)

#             node.add_child(child)

#             node_from_dict(node=child, Node=Node, rng=rng, dict=dict, loc=loc + str(i) + ',', **params)

# ------------------------------------------------------------ #
#                   Get Tree Info
# ------------------------------------------------------------ #

    def is_leaf(self, subtree=None):
        """Check if the current node is a leaf (has no children)."""

        subtree = self.tree if subtree is None else subtree

        if len(subtree) == 1:

            return True

        else:

            return False


    def is_constant(self, subtree):
        """Check if the current node is a constant. If the tree has been constructed
        correctly, this would also indicate that the node is a leaf. (However, not all
        leaves are constant.)"""

        if type(subtree[0]) == str or type(subtree[0]) == np.str_:

            if subtree[0][0] == 'c' and subtree[0][1].isdigit():

                return True

            else:

                return False

        else:

            return True


    def is_variable(self, subtree):
        """Check if current node is a variable"""

        if type(subtree[0]) == str or type(subtree[0]) == np.str_:

            if subtree[0] == 'x':

                return True

            elif len(subtree[0]) == 2:

                if subtree[0][0] == 'x' and subtree[0][1].isdigit():

                    return True

        return False


    def select_subtree(self, child_index_list, subtree=None):
        """Find the node associated with the index list and return it.

        Parameters:
            child_index_list: List is the indices of children. For example,
            the index_list of the 0-th child
            of node would be [0].
            subtree: current subtree"""

        subtree = self.tree if subtree is None else subtree

        if child_index_list == [] or child_index_list == [''] or child_index_list == '':

            return subtree

        else:

            index = child_index_list[0] if type(subtree[0]) == list else child_index_list[0]+1

            return self.select_subtree(subtree=subtree[index], child_index_list=child_index_list[1:])


    def set_subtree(self, new_subtree, child_index_list, subtree=None):
        """Find and set the node referenced by child_index_list equal to new_node.

        Parameters:
            new_subtree: The subtree that will be placed in
            the desired location.

            child_index_list: The location of the node that where
            new_subtree should be placed. List is the indices of children.
            For example, the index_list of the 0-th child of
            node would be [0].  Root node is indicated by ['']

            subtree: current subtree"""

        subtree = self.tree if subtree is None else subtree

        if child_index_list == []:

            # [:] means the locations of stored values of subtree will be changed.
            # Thus, self.tree will be affected by this assignment.
            subtree[:] = new_subtree

        else:

            index = child_index_list[0] if type(subtree[0]) == list else child_index_list[0] + 1

            return self.set_subtree(new_subtree=new_subtree,
                                    subtree=subtree[index],
                                    child_index_list=child_index_list[1:])


    def get_node_list(self, prefix='', node_list=None, subtree=None):
        """Get a list of all nodes below self. Each node is represented by a list
         of child indices.
        The list [0, 1, 1] would refer to the 0-th child's 1-st child's 1-st child.
        The parameters are
        mostly for recursion, and in most instances need not be specified."""

        subtree = self.tree if subtree is None else subtree

        if node_list is None:

            node_list = ['']

        # if subtree is a single node
        if len(subtree) == 1:

            return node_list

        else:

            if prefix != '':
                prefix = prefix + ','

            for i, st in enumerate(subtree):

                if type(st) == list:
                    node_list.append(tuple(map(int, (prefix+str(i-1)).split(','))))
                    node_list.extend(self.get_node_list(prefix=prefix + str(i-1), node_list=[], subtree=st))

            return node_list


    def get_node_dict(self, prefix='', node_dict=None, subtree=None):
        """Get a dictionary of all nodes in below self. Each node is represented by
        a list of child indices.
        The list [0, 1, 1] would refer to the 0-th child's 1-st child's 1-st child.
        The parameters are
        mostly for recursion, and in most instances need not be specified."""

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:

            node_dict = {'': subtree[0]}

        else:

            node_dict[tuple(map(int, prefix.split(',')))] = subtree[0]

        if len(subtree) == 1:

            return node_dict

        else:

            if prefix != '':
                prefix = prefix + ','

            for i, st in enumerate(subtree):

                if type(st) == list:
                    self.get_node_dict(prefix=prefix + str(i - 1), subtree=st, node_dict=node_dict)

            return node_dict


    def get_node_map(self, loc='', node_dict=None, subtree=None):
        """Look through all nodes and record nodes in a dictionary.
        This dictionary will fill all location of node names in the tree.

        So, to get the locations of a nodes named '+', simply do

        node_map = get_node_map()
        plus_node_locations = node_map['+']"""

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:
            node_dict = {}

        if subtree[0] in node_dict:

            node_dict[subtree[0]] = node_dict[subtree[0]].union({tuple(map(int, loc.split(',')))})

        else:

            if loc == '':
                node_dict[subtree[0]] = {loc}  # set literal

            else:
                node_dict[subtree[0]] = {tuple(map(int, loc.split(',')))}  # set literal

        if len(subtree) == 1:

            return node_dict

        else:

            if loc != '':
                loc = loc + ','

            for i, st in enumerate(subtree):

                if type(st) == list:
                    self.get_node_map(loc=loc + str(i - 1), subtree=st, node_dict=node_dict)


            return node_dict


    def get_tree_size(self, subtree=None, tree_size=0):
        """Count the number of nodes in the tree and return it."""

        if subtree is None:
            subtree = self.tree

        tree_size += 1

        if len(subtree) == 1:
            return tree_size

        else:

            for st in subtree:

                if type(st) == list:
                    tree_size = self.get_tree_size(subtree=st, tree_size=tree_size)

            return tree_size


    def get_depth(self):
        """Find the deepest node in the tree and return the depth. This is done
        by looking at the list of all nodes."""

        node_list = self.get_node_list()

        if node_list == ['']:
            return 0

        node_depth = [len(node_str) for node_str in node_list]

        return max(node_depth)


    def draw_tree(self, save_loc):
        """Draw tree

        Parameters:
            save_loc: The image generated will be saved in the location specified."""

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
         tree with root node = to self."""

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

        child, value = rule

        child_act = self.get_lisp_string(subtree[1])

        if child == child_act:

            subtree[:] = value
            count += 1

        return count


    def handle_simplification_rules2(self, rule, count, subtree):

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
        """Apply the simple simplification rules involving only x, 0, and 1."""

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

        In interval arithmetic, x*x is different than x^n.
        This simplification can avoid the problem."""

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
        does not check if this is an acceptable substitution."""

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

def get_parent(loc):
    """Get location of parent of loc."""

    if ',' in loc:

        return loc[:-2]

    else:

        return ''


def find_youngest_common_elder(l1, l2):
    """Find the youngest common node of loc1 and loc2."""

    # make loc2 the deeper location
    loc1, loc2 = (l2, l1) if len(l2) < len(l1) else (l1, l2)

    # find youngest common elder of loc1 and loc2
    common_elder = get_parent(loc1)

    while True:

        # we know loc1 if child of common elder
        # so only check loc2
        if is_elder(child=loc2, elder=common_elder):

            break

        else:

            common_elder = get_parent(common_elder)

    return common_elder


def is_consistent_operation_between(elder, child, op, node_map):
    """Check if operation is used at all nodes between child and elder
    including elder but not child.

    elder and child are locations

    op a string representing the operation (Ex. '*' for multiply)

    node_mape = self.get_node_map()"""

    # check if op is in tree at all
    if op not in node_map:

        return False

    # otherwise ...
    current_node = child

    while current_node != elder:

        current_node = get_parent(current_node)

        if current_node not in node_map[op]:

            return False

    return True


def get_other_child(loc):

    last_digit = '1' if loc[-1] == '0' else '0'

    other_child = loc[:-1] + last_digit

    return other_child


def is_elder(child, elder):
    """Returns True if child is lower on the tree (in the same branch) than elder.
    Both child and elder are strings representing the location of the node.
    The string is a sequence of child indices.

    Example: '0,0' is the 0th child of the 0th child of the root node."""

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


def is_same_branch(node1, node2):

    return is_elder(child=node1, elder=node2) or is_elder(child=node2, elder=node1)


def first_key(ordered_dictionary):
    """Get first element of an ordered dictionary"""

    return next(iter(ordered_dictionary))


def first_value(ordered_dictionary):

    return ordered_dictionary[first_key(ordered_dictionary)]


def get_branches(nodes):
    """Given a dictionary of at least a portion of tree, split
    the tree in it branches. If the entire tree is present, there
    should be only one branch. Return a list of branches, even
    if only one exists."""

    ordered_nodes = list(sorted(nodes.items(), key=lambda x: len(x[0])))
    branches = [collections.OrderedDict({ordered_nodes[0][0]: ordered_nodes[0][1]})]

    for loc, name in ordered_nodes[1:]:

        for branch in branches:

            if is_elder(elder=first_key(branch), child=loc):

                branch[loc] = name
                break

        else:

            branches.append(collections.OrderedDict({loc: name}))

    return branches


def count_consts(nodes):
    """Count how many different constants appear in the nodes."""

    counts = {}

    for key in nodes:

        if 'c' in nodes[key] and 'cos' != nodes[key]:

            if key in counts:

                counts[nodes[key]] += 1

            else:

                counts[nodes[key]] = 1

    return counts
