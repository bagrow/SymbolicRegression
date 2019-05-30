from GeneticProgrammingAfpo.protected_functions import *

import numpy as np

population_size = 100
max_generations = 100
max_front_size = int(np.sqrt(population_size))
max_depth = 6

# Functions from mainly from
# "Genetic Programming Needs Better Benchmarks" 2012.
function_dict = {'Koza-1': {'f': lambda x: x**4 + x**3 + x**2 + x,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 1},

                 'Koza-2': {'f': lambda x: x**5 - 2 * x**3 + x,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 2},

                 'Koza-3': {'f': lambda x: x**6 - 2 * x**4 + x**2,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 3},

                 'Nguyen-1': {'f': lambda x: x**3 + x**2 + x,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 4},

                 'Nguyen-3': {'f': lambda x: x**5 + x**4 + x**3 + x**2 + x,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 5},

                 'Nguyen-4': {'f': lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 6},

                 'Nguyen-5': {'f': lambda x: np.sin(x**2) * np.cos(x) - 1,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 7},

                 'Nguyen-6': {'f': lambda x: np.sin(x) + np.sin(x + x**2),
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 8},

                 'Nguyen-7': {'f': lambda x: plog(x + 1) + plog(x**2 + 1),
                              'type': 'urand',
                              'a': 0,
                              'b': 2,
                              'size': 20,
                              'factor': 9},

                 'Nguyen-8': {'f': lambda x: psqrt(x),
                              'type': 'urand',
                              'a': 0,
                              'b': 4,
                              'size': 20,
                              'factor': 10},

                 'Keijzer-1': {'f': lambda x: 0.3 * x * np.sin(2 * np.pi * x),
                               'type': 'uniform',
                               'a': -1,
                               'b': 1,
                               'step': 0.1,
                               'factor': 0},

                 'Keijzer-2': {'f': lambda x: 0.3 * x * np.sin(2 * np.pi * x),
                               'type': 'uniform',
                               'a': -2,
                               'b': 2,
                               'step': 0.1,
                               'factor': 11},

                 'Keijzer-4': {'f': lambda x: x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1),
                               'type': 'uniform',
                               'a': 0,
                               'b': 10,
                               'step': 0.05,
                               'factor': 12},

                 'scaled-sinc': {'f': lambda x: 5 * pdivide(np.sin(x), x),
                                 'type': 'urand',
                                 'a': 0,
                                 'b': 10,
                                 'size': 30},

                 'AutomaticFrenchCurve': {'f': lambda x: 4.26 * (np.exp(-x) - 4 * np.exp(-2 * x) + 3 * np.exp(-3 * x)),
                                          'type': 'urand',
                                          'a': 0,
                                          'b': 3.25,
                                          'size': 30},

                 'ChebyshevPolynomial': {'f': lambda x: 3 * np.cos(3 * np.arccos(x)),
                                         'type': 'urand',
                                         'a': -1,
                                         'b': 1,
                                         'size': 30},

                 'Ripple': {'f': lambda x: (x[0] - 3.) * (x[1] - 3.) + 2. * np.sin(x[0] - 4.) * (x[1] - 4.),
                            'type': 'urand',
                            'a': [0, 0],
                            'b': [5, 5],
                            'size': 30},

                 'RatPol3D': {'f': lambda x: 30. * (x[0] - 1.) * (x[2] - 1.) / x[1]**2 / (x[0] - 10.),
                              'type': 'urand',
                              'a': [0.05, 1, 0.05],
                              'b': [5, 2, 5],
                              'size': 30},   # per dimension

                 'UBall5D': {'f': lambda x: 10. / (5. + (x[0]-3.)**2 + (x[1]-3.)**2 + (x[2]-3.)**2 + (x[3]-3.)**2 + (x[4]-3.)**2),
                             'type': 'urand',
                             'a': [0.05]*5,
                             'b': [5.]*5,
                             'size': 30},

                 'identity': {'f': lambda x: x,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 30},

                 'Paige-1': {'f': lambda x: 1/(1+x[0]**(-4)) + 1/(1+x[1]**(-4)),
                             'type': 'urand',
                             'a': [-5]*2,
                             'b': [5]*2,
                             'size': 30},

                 'fr_test-1': {'f': lambda x: np.sin(pdivide(x[0], x[1], value=0.)),
                               'type': 'urand',
                               'a': [-1]*2,
                               'b': [1]*2,
                               'size': 30},

                 # ------------------------------------------------------------ #
                 #                         Datasets
                 # ------------------------------------------------------------ #

                 # Dataset must have a path to the dataset (excluding the file).
                 # This path begins with the location described by the environmental
                 # variable DATASET_PATH.
                 # The file must be called data.csv and the data must be formatted
                 # so that the output data (values to be predicted) are in the zero-th
                 # column the first input variable is in the first column ...

                 'combined_cycle_power_plant': {'path': 'uci_datasets/combined_cycle_power_plant_data_set'},

                 'wine': {'path': 'uci_datasets/wine'},

                 'airfoil_self_noise': {'path': 'uci_datasets/airfoil_self_noise_dataset'},

                 'auto_mpg': {'path': 'uci_datasets/auto_mpg_dataset'},
                 }

# Dictionary explaining how many children (inputs)
# are needed for each function.
required_children = {'*': 2,
                     '+': 2,
                     '-': 2,
                     '/': 2,
                     'p/': 2,
                     '%': 2,
                     'AQ': 2,
                     'abs': 1,
                     'exp': 1,
                     'sin': 1,
                     'cos': 1,
                     'psqrt': 1,
                     'plog': 1,
                     'parccos': 1,
                     'unary_minus': 1}

functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]

# and use this translation for function creation.
math_translate = {'p/': 'pdivide',
                  '%': 'pdivide',
                  'AQ': 'AQ',
                  'psqrt': 'psqrt',
                  '+': 'np.add',
                  '-': 'np.subtract',
                  '*': 'np.multiply',
                  '/': 'np.divide',
                  'sin': 'np.sin',
                  'cos': 'np.cos',
                  'exp': 'np.exp',
                  'parccos': 'parccos',
                  'iparcos': 'iparcos',
                  'unary_minus': 'unary_minus'}

# This is another translation for use with pyinterval.
math_translate_interval_arithmetic = {'p/': 'operator.truediv',  # fake pd
                                      '%': 'operator.truediv',  # fake pd
                                      '+': 'interval.__add__',
                                      '-': 'interval.__sub__',
                                      '*': 'interval.__mul__',
                                      '/': 'interval.__truediv__',
                                      'sin': 'imath.sin',
                                      'cos': 'imath.cos',
                                      'parccos': 'iparccos',
                                      'exp': 'imath.exp',
                                      # convert all constants to intervals
                                      '#f': lambda x: 'interval('+x+')'}

# rules for simplification for key: (a, b, c) where key is the function, a, b are the children and c is the value of the node
# & is used to prepresent anything (if two are used then the anythings must be equal.)
# In lisp notation this is (key a b) = c
simplification_rules = {'p/': (('&', '&', '1'), ('&', '(0)', '1'), ('(0)', '&', '0'), ('&', '(1)', '&')),
                        '%': (('&', '&', '1'), ('&', '(0)', '1'), ('(0)', '&', '0'), ('&', '(1)', '&')),
                        '/': (('(0)', '&', '0'), ('&', '(1)', '&')),
                        'AQ': (('&', '(0)', '&'), ('(0)', '&', '0')),
                        '-': (('&', '&', '0'), ('&', '(0)', '&')),
                        '*': (('(1)', '&', '&'), ('&', '(1)', '&'), ('(0)', '&', '0'), ('&', '(0)', '0')),
                        '+': (('(0)', '&', '&'), ('&', '(0)', '&')),
                        'psqrt': (('(0)', '0'), ('(1)', '1')),
                        'plog': (('(1)', '0'),),
                        'sin': (('(0)', '0'),),
                        'cos': (('(0)', '1'),)}

# Below here I don't really want this in this file.
# I need a way to keep this with primitive_set_transition
# stuff but allow this version of GP to use it.

# Dictionary explaining how many children (inputs)
# are needed for each function.
required_children['AQs'] = 2
required_children['CD'] = 2
required_children['FD'] = 2
required_children['TD'] = 2
required_children['CDs'] = 2
required_children['FDs'] = 2
required_children['TDs'] = 2
required_children['pfunc1'] = 2
required_children['pfunc2'] = 2

functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]

# and use this translation for function creation.
math_translate['AQs'] = 'AQs'
math_translate['CD'] = 'CD'
math_translate['FD'] = 'FD'
math_translate['TD'] = 'TD'
math_translate['CDs'] = 'CDs'
math_translate['FDs'] = 'FDs'
math_translate['TDs'] = 'TDs'
math_translate['pfunc1'] = 'pfunc1'
math_translate['pfunc2'] = 'pfunc2'

# This is another translation for use with pyinterval.
math_translate_interval_arithmetic['CDs'] = 'CDs_interval'
