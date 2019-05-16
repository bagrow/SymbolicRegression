import unittest
import GeneticProgrammingAfpo as GP


def setup(expr):

    t = GP.Tree(None, num_vars=10)
    t.from_string(expr)
    print(t.get_lisp_string())

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


if __name__ == '__main__':

    unittest.main()
