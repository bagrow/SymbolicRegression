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


def AQ(x, y):
    """Analytic Quotient"""

    return np.divide(x, np.sqrt(np.power(y, 2) + 1))


# For testing
if __name__ == '__main__':

    import time

    runs = 100000
    result = []
    result2 = []

    rng = np.random.RandomState(0)

    start = time.time()
    for _ in range(runs):
        pass

    print((time.time() - start) / runs)

    # rng = np.random.RandomState(0)
    start = time.time()

    for _ in range(runs):
        pass

    print((time.time() - start) / runs)

    print(np.all(np.array(result) == np.array(result2)))
