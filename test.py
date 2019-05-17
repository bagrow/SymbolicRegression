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


class TestIndividual(unittest.TestCase):

    def test_generate_individual_ptc2(self):

        I = GP.Individual(np.random.RandomState(0),
                          primitive_set=['*', '+', '-'],
                          terminal_set=['#x'])

        sizes = []

        for _ in range(100):
            I.generate_individual_ptc2(5, [1/5.]*5)
            sizes.append(I.get_tree_size())

        self.assertTrue(np.all(np.array(sizes) <= 5))


if __name__ == '__main__':

    unittest.main()
