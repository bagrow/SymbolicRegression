import numpy as np

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


# Below here I don't really want this in this file.
# I need a way to keep this with primitive_set_transition
# stuff but allow this version of GP to use it.
from interval import interval, inf, fpu


def pfunc1(f1, f2, s, x):
    """Parameterized function that transforms f1 into f2 as s goes from 1 to 0.
    Both f1 and f2 are expected to take only 1 argument."""

    return f1(x)*s + f2(x)*(1-s)


def pfunc2(f1, f2, s, x, y):
    """Parameterized function that transforms f1 into f2 as s goes from 1 to 0.
    Both f1 and f2 are expected to take 2 arguments."""

    return f1(x, y)*s + f2(x, y)*(1-s)


def AQs(s, x, y):
    """AQ with parameter s. If s = 0, we need to worry about
    unprotected division, so pdivide is used to solve this
    problem."""

    return pdivide(x, np.sqrt(np.add(np.power(y, 2), s)))


# ------------------------------------------------------------ #
#                   Continuous Division
# ------------------------------------------------------------ #

def CD(x, y):
    """Continuous division with s=1"""

    return CDs(1., x, y)


def FD(x, y):
    """Continuous division with s=1"""

    return FDs(1., x, y)


def TD(x, y):
    """Continuous division with s=1"""

    return TDs(1., x, y)


def mid(X, s):
    """Middle section of continuous division. This is a cubic
    with parameter s."""

    return (2 * s ** 2 * X - X ** 3) / s ** 4


def CDs(s, x, y):
    """Parameterized Continuous Division"""

    try:
        middle = mid(y, s)
        middle[np.logical_or(s == 0, np.isnan(middle))] = 0.

    except TypeError:

        if y == 0 or np.isnan(middle):
            middle = 0.

    return x*((np.heaviside(np.abs(y)-s, 1))*pdivide(1., y) + np.heaviside(s-np.abs(y), 0)*middle)


def vectorized_copysign(x, y):

    return


def FDs(s, x, y):
    """Parameterized division like CDs but no continuous at 0.
    This version has horizontal lines between -s and s."""

    try:
        middle = np.copysign(1, y)*1./s
        middle[np.logical_or(s == 0, np.isnan(middle))] = 0.

    except TypeError:

        if y == 0 or np.isnan(middle):
            middle = 0.

    return x*((np.heaviside(np.abs(y)-s, 1))*pdivide(1., y) + np.heaviside(s-np.abs(y), 0)*middle)


def TDs(s, x, y):
    """Parameterized division like CDs but no continuous at 0.
    This version has lines tangent to the points where y=-s and y=s."""

    try:

        middle = np.multiply(np.divide(-1, np.power(s, 2)), y)+np.copysign(1, y)*2./s
        middle[np.logical_or(s == 0, np.isnan(middle))] = 0.

    except TypeError:

        if y == 0 or np.isnan(middle):
            middle = 0.

    return x*((np.heaviside(np.abs(y)-s, 1))*pdivide(1., y) + np.heaviside(s-np.abs(y), 0)*middle)

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
