import kexpressions as k

import networkx as nx

import itertools

def test_get_location_map():

    tree_lists = [[['x0']],
                  [['1.0']],
                  [['*'], ['x0', 'x0']],
                  [['+'], ['-', 'x0'], ['*', '+'], ['x0', 'x0', 'x0', 'x0']]
                 ]
    answers = [{0: ()},
               {0: ()},
               {0: (), 1: (0,), 2: (1,)},
               {0: (), 1: (0,), 2: (1,), 3: (0,0), 4: (0,1), 5: (0,0,0), 6: (0,0,1), 7: (0,1,0), 8: (0,1,1)}]

    for i, (ans, tree_list) in enumerate(zip(answers, tree_lists)):
        gene = list(itertools.chain(*tree_list))

        output = k.get_location_map(gene, tree_list)
        yield check_output, ans, output, 'Failure on '+str(i)

def check_output(ans, output, msg):
    assert ans == output, msg


def test_build_tree():

    genes = [['x0'],
             ['1.0'],
             ['*', 'x0', 'x0'],
             ['+', '-', 'x0', '*', '+', 'x0', 'x0', 'x0', 'x0']
            ]

    answers = ['x0',
               '1.0',
               '(* x0 x0)',
               '(+ (- (* x0 x0) (+ x0 x0)) x0)']

    for i, (ans, gene) in enumerate(zip(answers, genes)):
        lisp, short_gene = k.build_tree(gene, return_short_gene=True)

        yield check_output, ans, lisp, 'Failure on '+str(i)


def test_to_s_expression():

    T1 = nx.Graph()
    T1.add_node((), label='x0')

    Thalf = nx.Graph()
    Thalf.add_node((), label='1.0')

    T2 = nx.Graph()
    T2.add_node((), label='*')
    T2.add_node((0,), label='x0')
    T2.add_node((1,), label='x0')

    T2.add_edge((), (0,))
    T2.add_edge((), (1,))

    T3 = nx.Graph()
    T3.add_node((), label='+')
    T3.add_node((0,), label='-')
    T3.add_node((1,), label='x0')
    T3.add_node((0,0), label='*')
    T3.add_node((0,1), label='+')
    T3.add_node((0,0,0), label='x0')
    T3.add_node((0,0,1), label='x0')
    T3.add_node((0,1,0), label='x0')
    T3.add_node((0,1,1), label='x0')

    T3.add_edge((), (0,))
    T3.add_edge((), (1,))
    T3.add_edge((0,), (0,0))
    T3.add_edge((0,), (0,1))
    T3.add_edge((0,0), (0,0,0))
    T3.add_edge((0,0), (0,0,1))
    T3.add_edge((0,1), (0,1,0))
    T3.add_edge((0,1), (0,1,1))

    T4 = nx.Graph()
    T4.add_node((), label='+')
    T4.add_node((0,), label='+')
    T4.add_node((1,), label='+')
    T4.add_node((0,0), label='%')
    T4.add_node((0,1), label='+')
    T4.add_node((1,0), label='%')
    T4.add_node((1,1), label='+')
    T4.add_node((0,0,0), label='%')
    T4.add_node((0,0,1), label='%')
    T4.add_node((0,1,0), label='%')
    T4.add_node((0,1,1), label='%')
    T4.add_node((1,0,0), label='%')
    T4.add_node((1,0,1), label='%')
    T4.add_node((1,1,0), label='%')
    T4.add_node((1,1,1), label='%')
    T4.add_node((0,0,0,0), label='x0')
    T4.add_node((0,0,0,1), label='x0')
    T4.add_node((0,0,1,0), label='x0')
    T4.add_node((0,0,1,1), label='x0')
    T4.add_node((0,1,0,0), label='x0')
    T4.add_node((0,1,0,1), label='x0')
    T4.add_node((0,1,1,0), label='x0')
    T4.add_node((0,1,1,1), label='x0')
    T4.add_node((1,0,0,0), label='x0')
    T4.add_node((1,0,0,1), label='x0')
    T4.add_node((1,0,1,0), label='x0')
    T4.add_node((1,0,1,1), label='x0')
    T4.add_node((1,1,0,0), label='x0')
    T4.add_node((1,1,0,1), label='x0')
    T4.add_node((1,1,1,0), label='x0')
    T4.add_node((1,1,1,1), label='x0')

    T4.add_edge((), (0,))
    T4.add_edge((), (1,))
    T4.add_edge((0,), (0,0))
    T4.add_edge((0,), (0,1))
    T4.add_edge((1,), (1,0))
    T4.add_edge((1,), (1,1))
    T4.add_edge((0,0), (0,0,0))
    T4.add_edge((0,0), (0,0,1))
    T4.add_edge((0,1), (0,1,0))
    T4.add_edge((0,1), (0,1,1))
    T4.add_edge((1,0), (1,0,0))
    T4.add_edge((1,0), (1,0,1))
    T4.add_edge((1,1), (1,1,0))
    T4.add_edge((1,1), (1,1,1))
    T4.add_edge((0,0,0), (0,0,0,0))
    T4.add_edge((0,0,0), (0,0,0,1))
    T4.add_edge((0,0,1), (0,0,1,0))
    T4.add_edge((0,0,1), (0,0,1,1))
    T4.add_edge((0,1,0), (0,1,0,0))
    T4.add_edge((0,1,0), (0,1,0,1))
    T4.add_edge((0,1,1), (0,1,1,0))
    T4.add_edge((0,1,1), (0,1,1,1))
    T4.add_edge((1,0,0), (1,0,0,0))
    T4.add_edge((1,0,0), (1,0,0,1))
    T4.add_edge((1,0,1), (1,0,1,0))
    T4.add_edge((1,0,1), (1,0,1,1))
    T4.add_edge((1,1,0), (1,1,0,0))
    T4.add_edge((1,1,0), (1,1,0,1))
    T4.add_edge((1,1,1), (1,1,1,0))
    T4.add_edge((1,1,1), (1,1,1,1))

    answers = [['x0'],
               ['1.0'],
               ['*', ['x0'], ['x0']],
               ['+', ['-', ['*', ['x0'], ['x0']], ['+', ['x0'], ['x0']]], ['x0']],
               ['+', ['+', ['%', ['%', ['x0'], ['x0']], ['%', ['x0'], ['x0']]], ['+', ['%', ['x0'], ['x0']], ['%', ['x0'], ['x0']]]], ['+', ['%', ['%', ['x0'], ['x0']], ['*', ['x0'], ['x0']]], ['+', ['%', ['x0'], ['x0']], ['%', ['x0'], ['x0']]]]]
              ]

    for ans, T in zip(answers, [T1, Thalf, T2, T3]):
        output = k.to_s_expression(T, ())

        yield check_output, output, ans, ''
