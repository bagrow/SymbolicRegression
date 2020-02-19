import GeneticProgramming as GP

import numpy as np


# ----------------------------------------------------------
# Tests of Tree Class
# ----------------------------------------------------------

# def setup(expr):

#     t = GP.Tree(None, num_vars=10)
#     print(expr)
#     t.from_string(expr)

#     return t


def test_get_tree_size():
    """Test that get_tree_size works"""
    print('test_get_tree_size')
    tree = GP.Tree(tree='(% (3) (x0))', actual_lisp=False)
    result = tree.get_tree_size()
    assert result == 3


def test_get_tree_size_special_counts():
    """Test that get_tree_size works with
    optional special_counts argument"""

    special_counts = {'%': 8,
                      'AQ': 8}

    tree = GP.Tree('(% (x0) (% (8) (2)))', actual_lisp=False)
    result = tree.get_tree_size(special_counts=special_counts)
    assert result == 19

def test_get_tree_size_special_counts2():
    """Test that get_tree_size works with
    optional special_counts argument"""

    special_counts = {'%': 8,
                      'AQ': 8}

    tree = GP.Tree('(% (x0) (AQ (8) (2)))')
    result = tree.get_tree_size(special_counts=special_counts)
    assert result == 19


def test_set_subtree_list():
    """Test that set_subtree works if
    child_index_list is specified by list"""

    tree = GP.Tree('(* (x0) (* (x0) (x0)))', actual_lisp=False)
    tree2 = GP.Tree('(- (x0) (3))', actual_lisp=False)
    tree.set_subtree(new_subtree=tree2.tree,
                     child_indices=[1, 0])
    assert tree.get_lisp_string() == '(* (x0) (* (- (x0) (3)) (x0)))'


def test_set_subtree_tuple():
    """Test that set_subtree works if
    child_index_list is specified by tuple"""

    tree = GP.Tree('(* (x0) (* (x0) (x0)))', actual_lisp=False)
    tree2 = GP.Tree('(- (x0) (3))', actual_lisp=False)
    tree.set_subtree(new_subtree=tree2.tree,
                     child_indices=(1, 0))
    assert tree.get_lisp_string() == '(* (x0) (* (- (x0) (3)) (x0)))'

def test_select_subtree_list():
    """Test that select_subtree works if
    child_index_list is specified by list"""

    tree = GP.Tree('(* (x0) (* (x1) (x2)))', actual_lisp=False, num_vars=3)
    subtree = tree.select_subtree(child_indices=[1, 0])
    assert subtree == ['x1']


def test_select_subtree_tuple():
    """Test that select_subtree works if
    child_index_list is specified by tuple"""

    tree = GP.Tree('(* (x0) (* (x1) (x2)))', actual_lisp=False, num_vars=3)
    subtree = tree.select_subtree(child_indices=(1, 0))
    assert subtree == ['x1']


def test_convert_to_standard():
    """Test if this works for a single
    constant."""

    tree = GP.Tree('(0)', actual_lisp=False)
    standard = tree.convert_lisp_to_standard(None)
    assert standard == '0+0*x[0]'


def test_convert_to_standard_10():
    """Test if this works for x_n where
    n >= 10."""

    tree = GP.Tree('(x143)', num_vars=200)
    standard = tree.convert_lisp_to_standard(None)
    assert standard == 'x[143]'

def test_convert_to_standard_10_more_nodes():
    """Test if this works for x_n where
    n >= 10 for a tree with more than one node."""

    tree = GP.Tree('(* x143 c3)', num_vars=200, actual_lisp=True)
    standard = tree.convert_lisp_to_standard({'*': 'Mult'})
    assert standard == 'Mult(x[143],c[3])'

def test_convert_to_standard_exponents():
    """Test if this works for x_n where
    n >= 10 for a tree with more than one node."""

    tree_list = ['*', ['x[0]**3'], [7]]
    tree = GP.Tree(tree_list)
    standard = tree.convert_lisp_to_standard({'*': 'Mult'})
    assert standard == 'Mult(x[0]**3,7)'


def test_convert_to_standard_exponents_one_node():
    """Test if this works for x_n where
    n >= 10 for a tree with more than one node."""

    tree_list = ['x[0]**2']
    tree = GP.Tree(tree_list)
    standard = tree.convert_lisp_to_standard({'*': 'Mult'})
    assert standard == 'x[0]**2'

def test_get_parent():

    loc = (1, 0)

    parent = GP.Tree.get_parent(loc)

    assert parent == (1,)


def test_get_num_leaves():

    tree = GP.Tree('(- (3) (+ (x0) (3)))')
    counts = tree.get_num_leaves(return_num_nodes=True)

    assert counts == (3, 5)

# ----------------------------------------------------------
# Tests of Individual Class
# ----------------------------------------------------------

def setup_individual():

    I = GP.Individual(np.random.RandomState(0),
                      primitive_set=['*', '+', '-'],
                      terminal_set=['#x'],
                      AFSPO=False)

    return I


def test_generate_individual_full():

    I = setup_individual()

    sizes = []

    for _ in range(100):
        I.tree = I.generate_individual_full(5)
        sizes.append(I.get_depth())

    assert np.all(np.array(sizes) == 5)


def test_generate_individual_grow():

    I = setup_individual()

    sizes = []

    for _ in range(100):
        I.tree = I.generate_individual_grow(5)
        sizes.append(I.get_depth())

    assert np.all(np.array(sizes) <= 5)


def test_generate_individual_ptc2():

    I = setup_individual()

    sizes = []

    for _ in range(100):
        I.generate_individual_ptc2(5, [1/5.]*5)
        sizes.append(I.get_tree_size())

    assert np.all(np.array(sizes) <= 5)


def test_dominates_same_ind():
    """Test if newer of two identical
    individuals dominates."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = I2.fitness = np.array([1., 5.])

    assert not I1.dominates(I2) and I2.dominates(I1)

def test_dominates_not_same_dominates():
    """Test dominates with different individuals."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = np.array([1., 4.])
    I2.fitness = np.array([1., 5.])

    assert I1.dominates(I2) and not I2.dominates(I1)

def test_dominates_not_same_dominates2():
    """Test dominates with different individuals
    (better in all)."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = np.array([.9, 4.])
    I2.fitness = np.array([1., 5.])

    assert I1.dominates(I2) and not I2.dominates(I1)

def test_dominates_not_same_non_dominated():
    """Test dominates with different individuals
    (non-dominated)."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = np.array([1., 4.])
    I2.fitness = np.array([.9, 5.])

    assert not I1.dominates(I2) and not I2.dominates(I1)


def test_dominates_three_objectives():
    """Test dominates with different individuals
    that have three objectives."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = np.array([.9, 4., 7.])
    I2.fitness = np.array([1., 5., 7.])

    assert I1.dominates(I2) and not I2.dominates(I1)

def test_dominates_three_objectives_non_dominate():
    """Test dominates with different individuals
    that have three objectives."""

    I1 = setup_individual()
    I2 = setup_individual()

    I1.fitness = np.array([.9, 4., 7.])
    I2.fitness = np.array([1., 5., 6.])

    assert not I1.dominates(I2) and not I2.dominates(I1)
