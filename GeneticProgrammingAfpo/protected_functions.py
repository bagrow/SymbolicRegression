import GeneticProgrammingAfpo.pickling as pickling

import numpy as np
from interval import interval, inf

import os

# ------------------------------------------------------------ #
#                     Protected Function
# ------------------------------------------------------------ #
# These are functions that don't incude all read numbers in
# their domain, so they are extended so that they will not
# cause errors.

def pdivide(x, y):
    """Protected divide function. This functions returns 1 when a
        denominator of 0 is input."""

    ans = np.true_divide(x, y)

    try:
        ans[np.logical_or(y == 0, np.isnan(ans))] = 1.

    except TypeError:

        if y == 0 or np.isnan(ans):
            ans = 1.

    return ans


def pdivide_no_numpy(x, y):

    try:
        return x / y

    except ZeroDivisionError:
        return 1.


def psqrt(x):
    """Protected square root function. This function returns the square root of
    the absolute value of x."""

    return np.sqrt(np.abs(x))


def plog(x):
    """Protected natural log function."""

    return np.log(np.abs(x))


def parccos(x):
    """Protected arccos. Inside [-1,1], use normal arccos.
    For x > 1, use arcos(1) = 0. For, x < -1 use arccos(-1) = pi."""

    ans = np.arccos(x)

    try:
        ans[x > 1.] = 0.
    except TypeError:
        if x > 1.:
            ans = 0.

    try:
        ans[x < -1.] = np.pi
    except TypeError:
        if x < -1.:
            ans = np.pi

    return ans


def AQ(x, y):
    """Analytic Quotient"""

    return np.divide(x, np.sqrt(np.power(y, 2) + 1))


def unary_minus(x):
    """Effectively a negative sign"""

    return np.negative(x)

# ------------------------------------------------------------ #
#          Modified Function for Interval Arithmetic
# ------------------------------------------------------------ #

@interval.function
def iparccos(x):
    """This is protected inverse cosine. Inside [-1,1], use normal arccos.
    For x > 1, use arcos(1) = 0. For, x < -1 use arccos(-1) = pi."""

    if x.inf <= -1.0:
        higher = np.pi

    elif x.inf >= 1.0:
        return (0.0, 0.0),

    else:
        higher = np.arccos(x.inf)

    if x.sup >= 1.0:
        lower = 0.0

    elif x.sup <= -1.0:
        return (np.pi, np.pi),

    else:
        lower = np.arccos(x.sup)

    return (lower, higher),


functions = {'pdivide': pdivide,
             'pdivide_no_numpy': pdivide_no_numpy,
             'psqrt': psqrt,
             'plog': plog,
             'parccos': parccos,
             'AQ': AQ,
             'unary_minus': unary_minus,
             'iparcos': iparccos}

pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.dill')

def pickle_protected_functions():

    print('Writing protected_functions from GeneticProgrammingAfpo')

    # Pickle them all
    if not os.path.exists(os.path.dirname(pickle_path_backup)):
        os.makedirs(os.path.dirname(pickle_path_backup))

    # pickle it here
    pickling.pickle_this(functions, pickle_path_backup)

    # In future versions this one will be edited
    # because this on will be loaded.
    pickling.pickle_this(functions, pickle_path)


# change stuff from here if it exists
if os.path.isfile(pickle_path):

    try:
        functions = pickling.unpickle_this(pickle_path)
        print(functions)
        for key in functions:
            print(key)
            exec('global' + key + '= functions[key]')
        # print(CDs(1,0))

    except ModuleNotFoundError:
        print('ModuleNotFoundError: Remaking GeneticProgrammingAfpo_protected_functions.dill')
        pickle_protected_functions()

# if pickled_path does not exist
else:
    pickle_protected_functions()
