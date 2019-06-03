from GeneticProgrammingAfpo.protected_functions import *
import GeneticProgrammingAfpo.pickling as pickling

import numpy as np

import os


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.pickle')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts_backup.pickle')

population_size = 100
max_generations = 100

def koza1(x): return x**4 + x**3 + x**2 + x
def koza2(x): return x**5 - 2 * x**3 + x
def koza3(x): return x**6 - 2 * x**4 + x**2

def nguyen1(x): return x**3 + x**2 + x
def nguyen3(x): return x**5 + x**4 + x**3 + x**2 + x
def nguyen4(x): return x**6 + x**5 + x**4 + x**3 + x**2 + x
def nguyen5(x): return np.sin(x**2) * np.cos(x) - 1
def nguyen6(x): return np.sin(x) + np.sin(x + x**2)
def nguyen7(x): return plog(x + 1) + plog(x**2 + 1)
def nguyen8(x): return psqrt(x)

def keijzer1(x): return 0.3 * x * np.sin(2 * np.pi * x)
def keijzer2(x): return 0.3 * x * np.sin(2 * np.pi * x)
def keijzer4(x): return x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1)

def scaled_sinc(x): return 5 * pdivide(np.sin(x), x)
def automatic_french_curve(x): return 4.26 * (np.exp(-x) - 4 * np.exp(-2 * x) + 3 * np.exp(-3 * x))
def chebyshev_polynomial(x): return 3 * np.cos(3 * np.arccos(x))
def ripple(x): return (x[0] - 3.) * (x[1] - 3.) + 2. * np.sin(x[0] - 4.) * (x[1] - 4.)
def rat_pol_3d(x): return 30. * (x[0] - 1.) * (x[2] - 1.) / x[1]**2 / (x[0] - 10.)
def u_ball_5d(x): return 10. / (5. + (x[0]-3.)**2 + (x[1]-3.)**2 + (x[2]-3.)**2 + (x[3]-3.)**2 + (x[4]-3.)**2)
def identity(x): return x
def paige1(x): return 1/(1+x[0]**(-4)) + 1/(1+x[1]**(-4))
def fr_test1(x): return np.sin(pdivide(x[0], x[1], value=0.))


# Functions from mainly from
# "Genetic Programming Needs Better Benchmarks" 2012.
function_dict = {'Koza-1': {'f': koza1,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 1},

                 'Koza-2': {'f': koza2,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 2},

                 'Koza-3': {'f': koza3,
                            'type': 'urand',
                            'a': -1,
                            'b': 1,
                            'size': 20,
                            'factor': 3},

                 'Nguyen-1': {'f': nguyen1,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 4},

                 'Nguyen-3': {'f': nguyen3,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 5},

                 'Nguyen-4': {'f': nguyen4,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 6},

                 'Nguyen-5': {'f': nguyen5,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 7},

                 'Nguyen-6': {'f': nguyen6,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 20,
                              'factor': 8},

                 'Nguyen-7': {'f': nguyen7,
                              'type': 'urand',
                              'a': 0,
                              'b': 2,
                              'size': 20,
                              'factor': 9},

                 'Nguyen-8': {'f': nguyen8,
                              'type': 'urand',
                              'a': 0,
                              'b': 4,
                              'size': 20,
                              'factor': 10},

                 'Keijzer-1': {'f': keijzer1,
                               'type': 'uniform',
                               'a': -1,
                               'b': 1,
                               'step': 0.1,
                               'factor': 0},

                 'Keijzer-2': {'f': keijzer2,
                               'type': 'uniform',
                               'a': -2,
                               'b': 2,
                               'step': 0.1,
                               'factor': 11},

                 'Keijzer-4': {'f': keijzer4,
                               'type': 'uniform',
                               'a': 0,
                               'b': 10,
                               'step': 0.05,
                               'factor': 12},

                 'scaled-sinc': {'f': scaled_sinc,
                                 'type': 'urand',
                                 'a': 0,
                                 'b': 10,
                                 'size': 30},

                 'AutomaticFrenchCurve': {'f': automatic_french_curve,
                                          'type': 'urand',
                                          'a': 0,
                                          'b': 3.25,
                                          'size': 30},

                 'ChebyshevPolynomial': {'f': chebyshev_polynomial,
                                         'type': 'urand',
                                         'a': -1,
                                         'b': 1,
                                         'size': 30},

                 'Ripple': {'f': ripple,
                            'type': 'urand',
                            'a': [0, 0],
                            'b': [5, 5],
                            'size': 30},

                 'RatPol3D': {'f': rat_pol_3d,
                              'type': 'urand',
                              'a': [0.05, 1, 0.05],
                              'b': [5, 2, 5],
                              'size': 30},   # per dimension

                 'UBall5D': {'f': u_ball_5d,
                             'type': 'urand',
                             'a': [0.05]*5,
                             'b': [5.]*5,
                             'size': 30},

                 'identity': {'f': identity,
                              'type': 'urand',
                              'a': -1,
                              'b': 1,
                              'size': 30},

                 'Paige-1': {'f': paige1,
                             'type': 'urand',
                             'a': [-5]*2,
                             'b': [5]*2,
                             'size': 30},

                 'fr_test-1': {'f': fr_test1,
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

# convert constants to intervals
def make_interval(const): return 'interval('+const+')'


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
                                      '#f': make_interval}

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


if not os.path.exists(os.path.dirname(pickle_path_backup)):
    os.makedirs(os.path.dirname(pickle_path_backup))

# pickle it here
pickling.pickle_this((population_size,
                      max_generations,
                      function_dict,
                      required_children,
                      math_translate,
                      math_translate_interval_arithmetic,
                      simplification_rules), pickle_path_backup)

# In future versions this one will be edited
# because this on will be loaded.
pickling.pickle_this((population_size,
                      max_generations,
                      function_dict,
                      required_children,
                      math_translate,
                      math_translate_interval_arithmetic,
                      simplification_rules), pickle_path)

functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]
