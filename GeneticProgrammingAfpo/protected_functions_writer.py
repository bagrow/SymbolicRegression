import GeneticProgrammingAfpo.pickling as pickling

import numpy as np
from interval import interval, inf

import os


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.pickle')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.pickle')

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


def intersection(I1, I2):

    I = interval()

    for a1, b1 in I1:

        for a2, b2 in I2:

            end_points = [a1, b1, a2, b2]
            indices = np.argsort(end_points)

            if np.all(indices[:2] == [0, 2]) or np.all(indices[:2] == [2, 0]):

                I = I | interval([end_points[indices[1]], end_points[indices[2]]])

            elif end_points[indices[1]] == end_points[indices[2]]:

                I = I | interval([end_points[indices[1]]])

    return I


def CDs_interval(s, x, y):

    mid = lambda X, param=s: (2*param**2*X-X**3)/param**4
    PD = lambda X: pdivide_no_numpy(1., X)

    return interval(x)*mid(intersection(interval(y), interval([-s, s]))) | interval(x)*PD(intersection(interval(y), interval([-inf, -s]))) | interval(x)*PD(intersection(interval(y), interval([s, inf])))


functions = {'pdivide': pdivide,
             'pdivide_no_numpy': pdivide_no_numpy,
             'psqrt': psqrt,
             'plog': plog,
             'parccos': parccos,
             'AQ': AQ,
             'unary_minus': unary_minus,
             'iparcos': iparccos}

# Pickle them all
if not os.path.exists(os.path.dirname(pickle_path_backup)):
    os.makedirs(os.path.dirname(pickle_path_backup))

# pickle it here
pickling.pickle_this(functions, pickle_path_backup)

# In future versions this one will be edited
# because this on will be loaded.
pickling.pickle_this(functions, pickle_path)
