import pickling_setup.consts_writer_fb
import pickling_setup.protected_functions_writer_fb
import GeneticProgramming.data_setup as ds
from GeneticProgramming.consts import *
import GeneticProgramming as GP
from es import evolutionary_strategy

import cma
import numpy as np
import pandas as pd

import argparse
import collections
import copy
import os


def update_constant(constant):
    """Get next constant.

    Parameters
    ----------
    constant : str
        A string the starts with c and ends
        with an integer. For now, I assume
        that there are only 10 constants.
        Thus, the constants are c0, c1, c2,
        ..., c9.
    """

    # num = [s.isdigit() for s in constant[1:]]

    return 'c'+str((int(constant[1])+1) % 10)


def get_one_hot_encoding(x, labels):
    """Get the one-hot encoding of x. If x is in
    primitives, the encoding refers to that list. If
    x is in terminals, the encoding refers to terminals.

    Perameters
    ----------
    x : str
        Value to be encoded.
    primitives : list
        All possible primitives.
    terminals : list
        All possible terminals.

    Returns
    -------
    encoding : list
        The one-hot encoding.
    """

    assert x in labels, 'x must be in labels'

    encoding = np.eye(len(labels))[labels.index(x)]

    return encoding


def test_get_one_hot_encoding():

    primitives = ['+', '-', '%', '*']

    encoding = get_one_hot_encoding('%', primitives)

    assert np.all(encoding == [0, 0, 1, 0])


def partially_fill_tree(pfill, ind):
    """Given partial fill (named pfill), update
    lisp.

    Parameters
    ----------
    pfill : dict
        A dictionary of labels for tree nodes
        where key=node location and value=label.
    lisp : str
        A lisp with only _p_ for primitives and
        _t_ for terminals.

    Returns
    -------
    order : dict
        Nodes to label. key=location, value=current label
    """

    for location in pfill:

        subtree = ind.select_subtree(child_indices=location)
        subtree[0] = pfill[location]

    # Get unlabeled nodes and their locations.
    order = ind.get_node_dict()

    for rm_key in [key for key in order if order[key] not in ('_p_', '_t_')]:
        del order[rm_key]

    return order


def get_network_input(tree, location, primitives, terminals, use_multiple_networks):
    """Get the input vector for the neural
    network.

    Parameters
    ----------
    tree : GP.Individual
        The tree that is being labeled.
    primitives : list
        The priitives set
    terminals : list
        The terminal set
    location : tuple
        The location of the node to be
        labeled.
    use_multiple_networks : bool
        If True, then one network use used
        to label a single node
        in the tree. Otherwise, additional
        input is needed.

    Returns
    -------
    x : np.array
        The input vector
    """

    vertex_dict = tree.get_node_dict()

    # order the dictionary
    # First consider then length of the vertex's location.
    # Then, consider the binary numbers made by the location.
    # If you have more than two children, this might cause problems.
    max_depth = max(map(len, vertex_dict.keys()))
    sort_function = lambda x: 2**(max_depth+1)*len(x[0])+sum([child*2**i for i, child in enumerate(x[0])])
    vertex_dict = collections.OrderedDict(sorted(vertex_dict.items(), key=sort_function))

    if use_multiple_networks:

        x = []

    else:

        # Get the depth of node to be labeled
        # and how far to the right of
        # left-most child (at that depth)
        # the current node is.
        depth = len(location)
        child = int(''.join(map(str, location)), 2)

        x = [depth, child]

    for loc in vertex_dict:

        # Get one of the labels to input.
        x_label = vertex_dict[loc]

        # We don't need the label
        # at location, so skip.
        if loc == location:

            if use_multiple_networks:
                continue

            else:

                num_zeros = len(primitives) if x_label in primitives or x_label == '_p_' else len(terminals)
                x = np.append(x, np.zeros(num_zeros))

        elif x_label in primitives:
            x = np.append(x, get_one_hot_encoding(x_label, primitives))

        elif x_label == '_p_':
            x = np.append(x, np.zeros(len(primitives)))

        elif x_label in terminals:
            x = np.append(x, get_one_hot_encoding(x_label, terminals))

        elif type(x_label) == np.float64:

            encoding = np.zeros(len(terminals))
            encoding[terminals.index('#f')] = 1

            x = np.append(x, encoding)

        elif x_label == '_t_':
            x = np.append(x, np.zeros(len(terminals)))

        else:
            print('What is going on? You shouldn\'t be here!')
            print('x_label', x_label, type(x_label))
            exit()

    return x


def label_node(tree, W, weight_dims, primitives, terminals, possible_output_labels,
               location, bookmark, constant, use_multiple_networks, hidden):
    """Label a single node.

    Parameters
    ----------
    tree : GP.Individual
        The tree that is being labeled.
    W : np.array
        The flattened weight matrices.
    weight_dims : 2D tuple
        The dimensions of current weight
        matrix. That is the one to label
        the node pointed to by location.
    primitives : list
        The priitives set
    terminals : list
        The terminal set
    possible_output_labels : list
        Contains all  possible labels that
        are output from the network. Possible
        the same as possible_input_labels.
    location : tuple
        The location of the node to be
        labeled.
    bookmark : int
        The current index in W.
    use_multiple_networks : bool
        If True, then one network use used
        to label a single node
        in the tree. Otherwise, additional
        input is needed.
    hidden : tuple
        A description of the hidden layers
        in the network. Every integer in the tuple
        specifies the number of nodes in that layer.
    """

    if use_multiple_networks:

        # w is a reshaped portion of W that is responsible
        # for labeling the current primitive
        w = W[bookmark:bookmark+np.prod(weight_dims)].reshape(weight_dims)

        if bookmark + np.prod(weight_dims) > len(W):

            print('bookmark too big')
            exit()

        # bookmark remembers where we are in W
        bookmark += np.prod(weight_dims)

    else:

        if hidden is not None:

            layer_sizes = (weight_dims[0], *hidden, weight_dims[-1])

            prev_size = layer_sizes[0]
            index_offset = 0
            w = []

            for size in layer_sizes[1:]:

                length = prev_size*size
                w.append(W[index_offset:index_offset+length].reshape((prev_size, size)))
                prev_size = size

        else:

            w = W.reshape(weight_dims)

    # Get the input to the network
    # x_label = tree.select_subtree(child_indices=GP.Tree.get_parent(location))[0]
    # x = get_one_hot_encoding(x_label, possible_input_labels)
    x = get_network_input(tree, location, primitives, terminals, use_multiple_networks)

    # Get the output of the network.
    if hidden is not None:

        output = np.matmul(x, w[0])

        for wi in w[1:]:
            output = np.matmul(np.tanh(output), wi)

    else:
        output = np.matmul(x, w)

    # If not using multiple networks the output
    # layer has both terminal outputs and primitive
    # outputs. We must ignore one of these types of
    # output based on which type of node is desired.
    if not use_multiple_networks:

        # If desired output is a primitive
        if primitives == possible_output_labels:

            start_output_indices = 0
            end_output_indices = len(primitives)


        else:

            start_output_indices = len(primitives)
            end_output_indices = len(primitives) + len(terminals)

            if '#f' in terminals:
                end_output_indices += 1

    else:

        start_output_indices = 0
        end_output_indices = len(possible_output_labels)

    # Get the index of the label based
    # on the largest network node in
    # the output layer. If #f is in
    # the outputs, check if it is selected
    # and if it is give number in last position.
    if '#f' in possible_output_labels:

        # Don't select the c-value output
        # unless #f is the largest
        output[:-1] = np.tanh(output[:-1])
        index = np.argmax(output[start_output_indices:end_output_indices-1])
        node_label = possible_output_labels[index]

        if node_label == '#f':
            node_label = output[-1]

    else:

        output = np.tanh(output)
        index = np.argmax(output[start_output_indices:end_output_indices])
        node_label = possible_output_labels[index]

        if node_label == '#c':

            # Use current placeholder constant
            # and then set next placeholder constant.
            node_label = constant
            constant = update_constant(constant)

    # Put the new label in the tree.
    subtree = tree.select_subtree(child_indices=location)
    subtree[0] = node_label

    return tree, bookmark


def label_tree(pfill, W, depth, primitives, terminals, use_multiple_networks, hidden):
    """Given the some existing labels (pfill),
    the depth of the full tree, and the weights of
    a neural network, label the remainder of the tree.

    Parameters
    ----------
    pfill : dict
        A dictionary of labels for tree nodes
        where key=node location and value=label.
    W : np.array
        A one dimensional array that represents
        the neural network. If a non-leaf node is being
        labeled, then it requires len(primitives) number
        of output nodes ortherwise it needs len(terminals).
        Weights are ordered like a lisp.
    depth : int
        The depth of the tree. This is used to
        reshape the weight array.
    primitives : list
        The priitives set
    terminals : list
        The terminal set
    hidden : tuple
        A description of the hidden layers
        in the network. Every integer in the tuple
        specifies the number of nodes in that layer.


    Returns
    -------
    ind : GP.Individual
        The labeled tree as an individual.
    """

    # Create an full tree of desired depth with _t_ for terminals
    # and _p_ for primitves. These labels will make it easy to
    # identify the accepible labels.
    ind = GP.Individual(rng, primitive_set=['_p_'],
                        terminal_set=['_t_'], method='full', depth=depth, IA=False)

    # Partially fill the tree. This
    # can be made more complicated, but for
    # now we just fill the root node. If
    # more complicated we will also have to
    # adjust order.
    order = partially_fill_tree(pfill, ind)

    # As we label each tree node, we will
    # use a portion of W. We use bookmark to
    # remember where we are in W.
    bookmark = 0

    # As constants are picked, we will put
    # placeholder constants in order starting
    # with c0.
    constant = 'c0'

    # len_terminals
    # is for the c-value node. This will provide the value of
    # the constant is #f is selected.
    len_terminals = len(terminals)+1 if '#f' in terminals else len(terminals)

    if not use_multiple_networks:

        # These will be the same for each node so no need to update them
        # inside the for loop.
        num_inputs = len(primitives)*(2**depth-1)+len(terminals)*2**depth+2
        num_outputs = len(primitives)+len_terminals

    # Now, fill the remainder of the tree
    # based on the pratial fill.
    for location in order:

        # if primitve
        if order[location] == '_p_':

            # the number of units in the network
            # that labels this primitive.
            if use_multiple_networks:

                num_inputs = len(primitives)*(2**depth-2)+len(terminals)*2**depth
                num_outputs = len(primitives)

            dims = (num_inputs, num_outputs)

            ind, bookmark = label_node(tree=ind, W=W, weight_dims=dims,
                                       primitives=primitives,
                                       terminals=terminals,
                                       possible_output_labels=primitives,
                                       location=location,
                                       bookmark=bookmark, constant=constant,
                                       use_multiple_networks=use_multiple_networks,
                                       hidden=hidden)

        # if terminal
        else:

            if use_multiple_networks:

                num_inputs = (len(primitives)+len(terminals))*(2**depth-1)
                num_outputs = len(primitives)

            dims = (num_inputs, num_outputs)

            ind, bookmark = label_node(tree=ind, W=W, weight_dims=dims,
                                       primitives=primitives,
                                       terminals=terminals,
                                       possible_output_labels=terminals,
                                       location=location,
                                       bookmark=bookmark, constant=constant,
                                       use_multiple_networks=use_multiple_networks,
                                       hidden=hidden)

    return ind


def get_partial_fills(primitives, locations, num_fills):
    """Get dictionary that will be used to fill the tree.

    Parameters
    ----------
    num_non_leaf : int
        The number of nodes that are not leaves (and not root).
        These are the nodes where primitives are placed.
    num_labels : int
        The number of primitives (or possible labels).
    num_fills : int
        The number of partial fills to create.

    Returns
    -------
    pfills : list
        A list of dictionaries describing the partial fill
        where key=location and value=label.
    """

    # k is the number of non-leaf nodes to fill.
    # m = len(primitives)
    n = len(locations)
    # p = [m**(k-1)*(m-1)/(m**(n)-1) for k in range(1, n+1)]

    pfills = [{(): p} for p in primitives]

    # This will happen if depth = 1
    if n == 0:
        return pfills

    terminal_depth = max(map(len, locations))

    # Give computed constants a number (like c0)
    # and keep track of the number to avoid unnecessary
    # duplicate. There are only 10 constants so if
    # consts_count gets to 10 repeats will begin.
    const_count = 0

    # Never pick a non-computed constant. We don't
    # want to be guessing the value of the const.
    sub_terminals = copy.copy(terminals)

    if '#f' in sub_terminals:
        sub_terminals.remove('#f')

    for _ in range(num_fills-len(primitives)):

        k = rng.choice(range(1, n))

        # Do this one uniformly and k times
        indices = rng.choice(n, size=k, replace=False)
        locs_subset = [locations[i] for i in indices]
        locs_labels = []

        for loc in locs_subset:

            if len(loc) == terminal_depth:

                # Never pick a non-computed constant. We don't
                # want to be guessing the value of the const.
                locs_labels.append(rng.choice(sub_terminals))

                # Pick an actual constant (for computed consts)
                # if that is what was picked.
                if locs_labels[-1] == '#c':

                    locs_labels[-1] = 'c%i' % (const_count % 10,)
                    const_count += 1

            else:
                locs_labels.append(rng.choice(primitives))


        pfill = {(): rng.choice(primitives)}

        for loc, label in zip(locs_subset, locs_labels):
            pfill[loc] = label

        pfills.append(pfill)

    return pfills


def f(W, dataset, depth, primitives, terminals, locations, use_multiple_networks, hidden, num_fills=10):

    pfills = get_partial_fills(primitives, locations, num_fills)
    train_fitnesses = []
    val_fitnesses = []

    for pfill in pfills:

        i = label_tree(pfill, W, depth, primitives, terminals, use_multiple_networks, hidden)

        i.evaluate_fitness(dataset, compute_val_error=True)

        train_fitnesses.append(i.fitness[0])
        val_fitnesses.append(i.validation_fitness)

    val_fitness = np.mean(val_fitnesses)

    global best

    if val_fitness < best[0]:

        best = (val_fitness, W)

    return np.mean(train_fitnesses)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # this will act as an index for rep_list (offset by 1 though)
    parser.add_argument('rep', help='Number of runs already performed', type=int)

    parser.add_argument('func', help='Specify the target function', type=str)
    parser.add_argument('exp', help='exp is the experiment number', type=int)

    parser.add_argument('-fe', '--function_evals', help='Maximum number of function evaluations.',
                        type=float, action='store', default=float('inf'))
    parser.add_argument('-npf', '--num_partial_fills', help='The number of partial fills to use in each evaluation',
                        action='store', type=int, default=6)
    parser.add_argument('-d', '--depth', help='The depth of the tree to be label',
                        action='store', type=int, default=3)
    parser.add_argument('-m', '--multiple_networks', help='If False, uses the same network to label each node.',
                        action='store_true')
    parser.add_argument('--hidden', help='The number of hidden nodes in each layer as a tuple.',
                        type=int, nargs='+', action='store')
    parser.add_argument('-t', '--timeout', help='Number of seconds after which to stop.',
                        action='store', type=float, default=float('inf'))

    parser.add_argument('--cmaes', help='Use cma-es', action='store_true')
    parser.add_argument('--nes', help='use natrual es', action='store_true')


    args = parser.parse_args()
    print(args)

    assert args.cmaes or args.nes, 'Either cmaes or nes must be True.'

    seed = args.rep + 1
    rng = np.random.RandomState(seed)

    base_path = os.path.join(os.environ['GP_DATA'],
                             'function_builder/experiments',
                             str(args.exp),
                             args.func)

    print('base_path', base_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    num_data_points = 200*100
    # frac_train = 0.7

    if args.func == 'test':

        target = lambda x: 2*x[0]

        X1 = np.linspace(-1, 1, num_data_points)
        X2 = np.linspace(-20, 3, num_data_points)
        rng.shuffle(X2)
        X = np.vstack((X1, X2))

        dataset = np.vstack((target(X), X1, X2)).T

    else:

        function_info = function_dict[args.func]

        X = np.array([rng.uniform(a, b, size=num_data_points) for a, b in zip(function_info['a'], function_info['b'])])

        target = function_info['f']

        dataset = np.vstack((target(X), X)).T

    k = 100
    folds = ds.get_k_folds(np.random.RandomState(0), k, dataset)

    dataset_train, dataset_test = ds.get_datasets_from_folds(args.rep, folds)

    val_frac = 0.2

    indices_val = rng.choice(len(dataset_train),
                             size=int(val_frac*len(dataset_train)),
                             replace=False)
    indices_train = [i for i in range(len(dataset_train)) if i not in indices_val]

    dataset_val = np.array([dataset_train[i] for i in indices_val])
    dataset_train = np.array([dataset_train[i] for i in indices_train])

    dataset_train_val = [dataset_train, dataset_val]

    depth = args.depth
    primitives = ['*', '+', '-', '%', 'sin2', 'id2']
    terminals = ['x0', 'x1', '#f']

    fake_individual = GP.Individual(rng=rng, primitive_set=primitives,
                                    terminal_set=terminals, method='full', depth=depth)

    node_dict = fake_individual.get_node_dict()
    # locations = [key for key in node_dict if node_dict[key] in primitives and key != ()]
    locations = list(node_dict.keys())
    locations.remove(())

    len_terminals = len(terminals)+1 if '#f' in terminals else len(terminals)

    # number_of_weights = ((2**depth)-2)*len(primitives)**2 + (2**depth)*len(primitives)*(len(terminals)+1)

    # The hidden layer structure as a tuple.
    # () = no hidden layers
    # (n,) =  one hidden layer with n nodes
    # (n, m) = two hidden layers, the first with n nodes and the second with m nodes
    # ...
    hidden = args.hidden

    if args.multiple_networks:
        number_of_weights = len(primitives)*((2**depth-2)*len(primitives) + 2**depth*len(terminals))*(2**depth-2)+len_terminals*(2**depth-1)*(len(primitives)+len(terminals))*2**depth

    else:

        if hidden is None:
            number_of_weights = ((2**depth-1)*len(primitives)+2**depth*len(terminals)+2)*(len(primitives)+len_terminals)

        else:

            number_of_weights = ((2**depth-1)*len(primitives)+2**depth*len(terminals)+2)*hidden[0]+hidden[-1]*(len(primitives)+len_terminals)

            if len(hidden) > 1:

                number_of_weights += sum([hidden[i]*hidden[i+1] for i in range(len(hidden)-1)])

    if args.cmaes:

        # best = (error, weights)
        best = (float('inf'), None)

        xopt, es = cma.fmin2(f, 2*rng.rand(number_of_weights)-1, 0.1,
                             args=(dataset_train_val, depth, primitives, terminals, locations, args.multiple_networks, hidden, args.num_partial_fills),
                             options={'maxfevals': args.function_evals,
                                      'ftarget': 1e-10,
                                      'seed': seed,
                                      'verb_log': 0,
                                      'timeout': args.timeout},
                             restarts=0)

        print('best validation error', best[0])
        xopt = best[1]

    elif args.nes:
        # xopt = 
        pass

    function_builder_data = []

    for pfill in [{(): '*'}, {(): '%'}, {(): '+'}, {(): '-'}, {(): 'id2'}, {(): 'sin2'}]:

        i = label_tree(pfill, xopt, depth, primitives, terminals, args.multiple_networks, hidden)

        i.evaluate_fitness(dataset_train_val, compute_val_error=True)

        i.evaluate_test_points(data=dataset_test)

        if hasattr(i, 'c'):
            print(i.c)

        lisp = i.get_lisp_string(actual_lisp=True)
        print(lisp)

        row = [lisp, i.fitness[0], i.testing_fitness]
        function_builder_data.append(row)

    filename_id = '_rep'+str(args.rep)

    df_weights = pd.DataFrame(xopt)
    df_weights.to_csv(os.path.join(base_path, 'weights'+filename_id+'.csv'), index=False, header=None)

    df_function_builder = pd.DataFrame(function_builder_data)
    df_function_builder.to_csv(os.path.join(base_path, 'function_builder_data'+filename_id+'.csv'), index=False,
                               header=['s-expression', 'Training Error', 'Testing Error'])
