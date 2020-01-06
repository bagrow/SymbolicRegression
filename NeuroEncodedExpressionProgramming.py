"""Encoding of trees in this algorithm are based on the
encoding in gene expression programming. The encodings of the
equations/programs are called k-expressions or karva notation.

https://www.gepsoft.com/gepsoft/APS3KB/Chapter05/Section3/SS1.htm
"""

from GeneticProgramming.consts import *
from GeneticProgramming.Individual import Individual

import cma
import numpy as np
import pandas as pd
import networkx as nx

import collections


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
    output = np.matmul(hidden, w)

    return output, hidden


def get_value_no_input(v, w, hidden, activation, no_activation_on_constant):
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
    no_activation_on_constant : bool
        If True, don't apply the activation function to
        the constant node which is the second to last one.

    Returns
    -------
    output : 1D np.array
        The value of the nodes in the output layer.
    hidden: 1D np.array
        The new values of the nodes in the hidden layer.
    """

    hidden = activation(np.matmul(hidden, v))

    if no_activation_on_constant:

        output = np.matmul(hidden, w)

        # Get all indices except second to last one
        # and apply the activation function to all those indices.
        none_const_indices = list(range(len(output)))
        del none_const_indices[-2]
        none_const_indices = np.array(none_const_indices)

        output[none_const_indices] = activation(output[none_const_indices])

    else:
        output = activation(np.matmul(hidden, w))

    return output, hidden


def get_tail_size(head_size, max_num_children):
    """Get the size of the tail based on the size
    of the head and the maximum number of child nodes
    required from the primitives.

    Parameters
    ----------
    head_size : int
        The number of elements in the head portion
        of the list.
    max_num_children : int
        The maximum number of children required
        by any allowed primitive.

    Results
    -------
    tail_size : int
        The number of elements in the tail portion
        of the list.
    """

    return head_size*(max_num_children-1)+1


def build_tree(gene):
    """Take list version of equation
    and make a tree."""

    # Reorganize gene into a list of lists
    # where each sublist is from the same layer
    # of the tree. Call this new list tree_list.
    tree_list = [[gene[0]]]
    tree_list_index = 0
    current_index = 1

    while current_index < len(gene):

        # Get the number of nodes in the next layer.
        num_next_layer = sum([required_children[label] for label in tree_list[tree_list_index] if label in required_children])

        # In case there are extra elements in gene,
        # check that the next level should actually
        # exist.
        if num_next_layer == 0:
            break

        tree_list.append(gene[current_index:current_index+num_next_layer])

        current_index += num_next_layer
        tree_list_index += 1


    def get_location_map(gene, tree_list, i=0, j=0, k=0,
                         prefix=(), locations={0: ()}):
        """Gets a dictionary that relates index of gene
        (list of tree labels) to the position of each
        label in the tree.

        Parameters
        ----------
        gene : list
            The equation as a list of labels. This
            is the typcial form in GEP.
        tree_list : list of lists
            This is the same elements of gene but grouped
            by layer in the tree. The 0th layer is only the root
            node.
        i : int
            index in gene (for recursion)
        j : int
            index in tree_list  (for recursion)
        k : int
            index in tree_list second level and offset
            for calculating new i (for recursion)
        prefix : tuple
            The location of the current node by child numbers.
            For example root is () and the leftmost child of
            root is (0,) because it is the zeroth child of root.
            And, (0,0) is the zeroth child of the zeroth child of
            root. (for recursion)
        locations : dict
            This will be the dictionary returned at the end. Each
            key corresponds to an index in gene (i) and the value
            is the location (prefix).

        Returns
        -------
        locations : dict
            This will be the dictionary returned at the end. Each
            key corresponds to an index in gene (i) and the value
            is the location (prefix).
        """

        # if primitive
        if gene[i] in required_children:

            # For each of the required child nodes of gene[i]
            for child_num in range(required_children[gene[i]]):

                # update k so that i has the correct offset
                new_k = i - sum(len(layer) for layer in tree_list[:j])
                new_k = sum(required_children[label] for label in tree_list[j][:new_k] if label in required_children)

                # update i to be the child of the current i
                new_i = sum(len(layer) for layer in tree_list[:j+1])+child_num+new_k

                get_location_map(gene, tree_list, i=new_i, j=j+1, k=new_k,
                                 prefix=(*prefix, child_num), locations=locations)

        # update the locations dictionary
        # with the current location.
        locations[i] = prefix

        return locations

    # Gets a dictionary that relates index of gene
    # (list of tree labels) to the position of each
    # label in the tree.
    locations = get_location_map(gene, tree_list)

    # Get a new dictionary that has keys of the locations
    # and values of the labels.
    tree_dict = collections.OrderedDict()

    for i in range(len(locations)):
        tree_dict[locations[i]] = gene[i]

    # Now create a tree with networkx.
    # This can probably be avoided by further modifying
    # the nx.to_nested_tuple() function.
    T = nx.Graph()

    for loc in tree_dict:
        T.add_node(loc, label=tree_dict[loc])

        if loc != ():
            T.add_edge(loc, loc[:-1])

    return to_s_expression(T, ())


def get_individual(rng, w, recursive_weights, num_hidden, num_outputs, primitives, terminals,
                   head_size, num_time_steps):

    gene = generate_equation(rng, w, recursive_weights, num_hidden, num_outputs,
                             primitives, terminals, head_size, num_time_steps)

    tree = build_tree(gene)

    ind = Individual(rng, primitives, terminals, num_vars=10,
                     tree=tree)

    return ind


def to_s_expression(T, root, canonical_form=False):
    """Modified version of nx.to_nested_tuple().

    Returns a nested tuple representation of the given tree.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    root : node
        The node in ``T`` to interpret as the root of the tree.

    canonical_form : bool
        If ``True``, each tuple is sorted so that the function returns
        a canonical form for rooted trees. This means "lighter" subtrees
        will appear as nested tuples before "heavier" subtrees. In this
        way, each isomorphic rooted tree has the same nested tuple
        representation.

    Returns
    -------
    tuple
        A nested tuple representation of the tree.
    """

    def _make_tuple(T, root, _parent, nested):
        """Recursively compute the nested tuple representation of the
        given rooted tree.

        ``_parent`` is the parent node of ``root`` in the supertree in
        which ``T`` is a subtree, or ``None`` if ``root`` is the root of
        the supertree. This argument is used to determine which
        neighbors of ``root`` are children and which is the parent.

        """
        # Get the neighbors of `root` that are not the parent node. We
        # are guaranteed that `root` is always in `T` by construction.

        children = set(T[root]) - {_parent}

        if len(children) == 0:
            return [T.node[root]['label']]

        nested = [T.node[root]['label']] + [_make_tuple(T, v, root, nested) for v in children]

        # if canonical_form:
        #     nested = sorted(nested)
        return nested

    # Do some sanity checks on the input.
    if not nx.is_tree(T):
        raise nx.NotATree('provided graph is not a tree')

    if root not in T:
        raise nx.NodeNotFound('Graph {} contains no node {}'.format(T, root))

    return _make_tuple(T, root, None, None)


def generate_equation(rng, w, recursive_weights, num_hidden, num_outputs,
                      primitives, terminals, head_size, num_time_steps):
    """Get sequence from NN that can be converted into a equation.

    Parameters
    ----------

    Returns
    -------
    gene : list of str
        The output sequence."""

    hidden_values = w[:num_hidden]

    output_weights = w[num_hidden:].reshape((num_hidden, num_outputs))

    guassian = lambda x: np.exp(-np.power(x, 2))

    n = max([required_children[p] for p in primitives])

    tail_size = head_size*(n-1) + 1

    output_list = primitives + terminals

    gene = [None]*(head_size+tail_size)
    L = 0

    no_activation_on_constant = True if '#f' in terminals else False

    while L < len(gene):

        for _ in range(num_time_steps):
            output, hidden_values = get_value_no_input(recursive_weights, output_weights,
                                                       hidden_values,
                                                       activation=guassian,
                                                       no_activation_on_constant=no_activation_on_constant)

        if L < head_size:

            # P = int(L*output[-1]+1.5)
            # L = max(L, P)

            if '#f' in terminals:

                label_index = np.argmax(output[:-2])

                if label_index == len(output) - 3:
                    label = output[-2]

                else:
                    label = output_list[label_index]

            else:

                label_index = np.argmax(output[:-1])
                label = output_list[label_index]


        else:

            # P = int(output[-1]*(L-head_size+1) + head_size + 0.5)
            # L = max(L, P)

            if '#f' in terminals:

                label_index = np.argmax(output[len(primitives):-2])

                if label_index == len(output) - 3:
                    label = output[-2]

                else:
                    label = terminals[label_index]

            else:

                label_index = np.argmax(output[:-1])
                label = terminals[label_index]

        P = L
        gene[P] = label
        L += 1
    print('gene', gene)
    return gene


def f(w, rng, recursive_weights, num_hidden, head_size, num_outputs,
      primitives, terminals, dataset, num_time_steps):

    ind = get_individual(rng, w, recursive_weights, num_hidden, num_outputs,
                         primitives, terminals, head_size, num_time_steps)

    ind.evaluate_fitness(dataset, compute_val_error=True)

    global best

    if ind.validation_fitness < best[0]:

        best = (ind.validation_fitness, w)

    return ind.fitness[0]


def run_neuro_encoded_expression_programming(rep, num_hidden, head_size, primitives,
                                             terminals, dataset, dataset_test,
                                             timeout, function_evals, base_path,
                                             num_time_steps, sigma=1):

    seed = rep + 1
    rng = np.random.RandomState(seed)

    recursive_weights = rng.uniform(-1, 1, size=(num_hidden, num_hidden))

    num_outputs = len(primitives) + len(terminals) + 1

    if '#f' in terminals:
        num_outputs += 1

    # Start with 50% of weights at 0.
    # The rest are given a random value in [-2, 2]
    w = np.zeros(num_outputs*num_hidden + num_hidden)
    indices = rng.choice(num_outputs*num_hidden, size=num_outputs*num_hidden//2)
    values = rng.uniform(-2, 2, size=num_outputs*num_hidden//2)
    w[indices] = values

    global best

    # best = (error, weights)
    best = (float('inf'), None)

    xopt, es = cma.fmin2(f, w, sigma,
                         args=(rng, recursive_weights, num_hidden,
                               head_size, num_outputs, primitives,
                               terminals, dataset, num_time_steps),
                         options={'maxfevals': function_evals,
                                  'popsize': 100,
                                  # 'ftarget': 1e-10,
                                  'tolfun': 0,
                                  # 'tolfunhist': 0,
                                  'seed': seed,
                                  'verb_log': 0,
                                  'timeout': timeout},
                         restarts=0)

    print('best validation error', best[0])
    xopt = best[1]

    filename_id = '_rep'+str(rep)

    i = get_individual(rng, xopt, recursive_weights, num_hidden, num_outputs,
                       primitives, terminals, head_size, num_time_steps)

    i.evaluate_fitness(dataset, compute_val_error=True)

    i.evaluate_test_points(data=dataset_test)

    if hasattr(i, 'c'):
        print(i.c)

    lisp = i.get_lisp_string(actual_lisp=True)
    print('equation', lisp)

    neep_data = [[lisp, i.fitness[0], i.validation_fitness, i.testing_fitness]]

    if base_path is not None:

        df_neep = pd.DataFrame(neep_data)
        df_neep.to_csv(os.path.join(base_path, 'neuro_encoded_expression_programming_data'+filename_id+'.csv'), index=False,
                       header=['s-expression', 'Training Error', 'Validation Error', 'Testing Error'])

        df_weights = pd.DataFrame(xopt)
        df_weights.to_csv(os.path.join(base_path, 'weights'+filename_id+'.csv'), index=False, header=None)

    return i.testing_fitness


if __name__ == '__main__':

    if False:

        # to use nn to output seq and then convert
        rng = np.random.RandomState(0)

        num_hidden = 40
        recursive_weights = rng.uniform(-1, 1, size=(num_hidden, num_hidden))

        primitives = ['*', '+', '-', '%']
        terminals = ['x0', 'x1', '#f']
        head_size = 30

        num_outputs = len(primitives) + len(terminals) + 1

        if '#f' in terminals:
            num_outputs += 1

        w = np.zeros(num_outputs*num_hidden + num_hidden)
        indices = rng.choice(num_outputs*num_hidden, size=num_outputs*num_hidden//2)
        values = rng.uniform(-2, 2, size=num_outputs*num_hidden//2)
        w[indices] = values
        num_time_steps = 2

        ind = get_individual(rng, w, recursive_weights, num_hidden, num_outputs,
                             primitives, terminals, head_size, num_time_steps)

        print(ind.get_lisp_string())

    else:

        # to just convert to tree...
        rng = np.random.RandomState(0)

        primitives = ['*', '+', '-', '%']
        terminals = ['x0', 'x1', '#f']

        gene = ['*', 'x0', 'x0', '*']

        tree = build_tree(gene)

        ind = Individual(rng, primitives, terminals, num_vars=1,
                         tree=tree)

        print(ind.get_lisp_string())
