"""Encoding of trees in this algorithm are based on the
encoding in gene expression programming. The encodings of the
equations/programs are called k-expressions or karva notation.

https://www.gepsoft.com/gepsoft/APS3KB/Chapter05/Section3/SS1.htm
"""

from GeneticProgramming.consts import *

import networkx as nx

import collections


def build_tree(gene, return_short_gene=False):
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

    lisp = to_s_expression(T, ())

    if return_short_gene:

        short_gene_end_index = sum([len(x) for x in tree_list])
        short_gene = gene[:short_gene_end_index]

        return lisp, short_gene
    
    else:
        return lisp


def to_s_expression(T, root, canonical_form=False):
    """Modified version of networkx.to_nested_tuple().

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
