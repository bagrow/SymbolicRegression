import unittest
import GeneticProgrammingAfpo as GP

import numpy as np


def setup(expr):

    t = GP.Tree(None, num_vars=10)
    t.from_string(expr)

    return t


class TestTree(unittest.TestCase):

    def test_get_tree_size(self):
        """Test that get_tree_size works"""

        tree = setup('(% (3) (x0))')
        result = tree.get_tree_size()
        self.assertEqual(result, 3)


    def test_get_tree_size_special_counts(self):
        """Test that get_tree_size works with
        optional special_counts argument"""

        special_counts = {'%': 8,
                          'AQ': 8}

        tree = setup('(% (x0) (% (8) (2)))')
        result = tree.get_tree_size(special_counts=special_counts)
        self.assertEqual(result, 19)

    def test_get_tree_size_special_counts2(self):
        """Test that get_tree_size works with
        optional special_counts argument"""

        special_counts = {'%': 8,
                          'AQ': 8}

        tree = setup('(% (x0) (AQ (8) (2)))')
        result = tree.get_tree_size(special_counts=special_counts)
        self.assertEqual(result, 19)


    def test_set_subtree_list(self):
        """Test that set_subtree works if
        child_index_list is specified by list"""

        tree = setup('(* (x0) (* (x0) (x0)))')
        tree2 = setup('(- (x0) (3))')
        tree.set_subtree(new_subtree=tree2.tree,
                         child_indices=[1, 0])
        self.assertEqual(tree.get_lisp_string(), '(* (x0) (* (- (x0) (3)) (x0)))')


    def test_set_subtree_tuple(self):
        """Test that set_subtree works if
        child_index_list is specified by tuple"""

        tree = setup('(* (x0) (* (x0) (x0)))')
        tree2 = setup('(- (x0) (3))')
        tree.set_subtree(new_subtree=tree2.tree,
                         child_indices=(1, 0))
        self.assertEqual(tree.get_lisp_string(), '(* (x0) (* (- (x0) (3)) (x0)))')

    def test_select_subtree_list(self):
        """Test that select_subtree works if
        child_index_list is specified by list"""

        tree = setup('(* (x0) (* (x1) (x2)))')
        subtree = tree.select_subtree(child_indices=[1, 0])
        self.assertEqual(subtree, ['x1'])


    def test_select_subtree_tuple(self):
        """Test that select_subtree works if
        child_index_list is specified by tuple"""

        tree = setup('(* (x0) (* (x1) (x2)))')
        subtree = tree.select_subtree(child_indices=(1, 0))
        self.assertEqual(subtree, ['x1'])


    def test_convert_to_standard(self):
        """Test if this works for a single
        constant."""

        tree = GP.Tree('(0)')
        standard = tree.convert_lisp_to_standard(None)
        self.assertEqual(standard, '0+0*x[0]')


    def test_convert_to_standard_10(self):
        """Test if this works for x_n where
        n >= 10."""

        tree = GP.Tree('(x143)', num_vars=200)
        standard = tree.convert_lisp_to_standard(None)
        self.assertEqual(standard, 'x[143]')

    def test_convert_to_standard_10_more_nodes(self):
        """Test if this works for x_n where
        n >= 10 for a tree with more than one node."""

        tree = GP.Tree('(* x143 c3)', num_vars=200, actual_lisp=True)
        standard = tree.convert_lisp_to_standard({'*': 'Mult'})
        self.assertEqual(standard, 'Mult(x[143],c[3])')

    def test_convert_to_standard_exponents(self):
        """Test if this works for x_n where
        n >= 10 for a tree with more than one node."""

        tree_list = ['*', ['x[0]**3'], [7]]
        tree = GP.Tree(tree_list)
        standard = tree.convert_lisp_to_standard({'*': 'Mult'})
        self.assertEqual(standard, 'Mult(x[0]**3,7)')


    def test_convert_to_standard_exponents_one_node(self):
        """Test if this works for x_n where
        n >= 10 for a tree with more than one node."""

        tree_list = ['x[0]**2']
        tree = GP.Tree(tree_list)
        standard = tree.convert_lisp_to_standard({'*': 'Mult'})
        self.assertEqual(standard, 'x[0]**2')


def setup_individual():

    I = GP.Individual(np.random.RandomState(0),
                      primitive_set=['*', '+', '-'],
                      terminal_set=['#x'])

    return I


class TestIndividual(unittest.TestCase):

    def test_generate_individual_full(self):

        I = setup_individual()

        sizes = []

        for _ in range(100):
            I.tree = I.generate_individual_full(5)
            sizes.append(I.get_depth())

        self.assertTrue(np.all(np.array(sizes) == 5))


    def test_generate_individual_grow(self):

        I = setup_individual()

        sizes = []

        for _ in range(100):
            I.tree = I.generate_individual_grow(5)
            sizes.append(I.get_depth())

        self.assertTrue(np.all(np.array(sizes) <= 5))


    def test_generate_individual_ptc2(self):

        I = setup_individual()

        sizes = []

        for _ in range(100):
            I.generate_individual_ptc2(5, [1/5.]*5)
            sizes.append(I.get_tree_size())

        self.assertTrue(np.all(np.array(sizes) <= 5))


    def test_dominates_same_ind(self):
        """Test if newer of two identical
        individuals dominates."""

        I1 = setup_individual()
        I2 = setup_individual()

        I1.fitness = I2.fitness = np.array([1., 5.])

        self.assertTrue(not I1.dominates(I2) and I2.dominates(I1))

    def test_dominates_not_same_dominates(self):
        """Test dominates with different individuals."""

        I1 = setup_individual()
        I2 = setup_individual()

        I1.fitness = np.array([1., 4.])
        I2.fitness = np.array([1., 5.])

        self.assertTrue(I1.dominates(I2) and not I2.dominates(I1))

    def test_dominates_not_same_dominates2(self):
        """Test dominates with different individuals
        (better in all)."""

        I1 = setup_individual()
        I2 = setup_individual()

        I1.fitness = np.array([.9, 4.])
        I2.fitness = np.array([1., 5.])

        self.assertTrue(I1.dominates(I2) and not I2.dominates(I1))

    def test_dominates_not_same_non_dominated(self):
        """Test dominates with different individuals
        (non-dominated)."""

        I1 = setup_individual()
        I2 = setup_individual()

        I1.fitness = np.array([1., 4.])
        I2.fitness = np.array([.9, 5.])

        self.assertTrue(not I1.dominates(I2) and not I2.dominates(I1))


if __name__ == '__main__':

    unittest.main()
