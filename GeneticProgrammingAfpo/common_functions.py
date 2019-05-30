# These two imports are absolutely necessary. I think it is because
# the lambda strings (which contain np functions) are eval-ed here.
from .protected_functions import *
import numpy as np
import operator

import copy

# ------------------------------------------------------------ #
#                   Function Creation
# ------------------------------------------------------------ #


def get_lambda_string(expr, const=False):
    """This function creates a lambda string.

    Parameters
    ----------
    expr : string
        The right-hand size of a mathematical function with
        input x and possibly input c. Note that x and/or c
        can be np.arrays.
    const : bool (optional)
        If true, lambda expression will use c as input.

    Returns
    -------
    lstr : string
        String with lambda at the begin. This string is
        formated such that eval(lstr) with work.

    Examples
    --------
    >>> expr = 'x[0]+4'
    >>> common_functions.get_lambda_string(expr)
    lambda x: x[0]+4

    >>> expr = 'x[0]*x[1]+c[0]
    >>> common_functions.get_lambda_string(expr, const=True)
    lambda x: x[0]*x[1]+c[0]
    """

    vars = 'x' if not const else 'c,x'

    lstr = 'lambda ' + vars + ': ' + expr

    return lstr


def get_function(expr, const=False):
    """Calls get_lambda_string() and then evaluates the results to create
    a lambda function of the expression (expr).

    Parameters
    ----------
    expr : string
        The right-hand size of a mathematical function with
        input x and possibly input c. Note that x and/or c
        can be np.arrays.
    const : bool (optional)
        If true, lambda expression will use c as input.

    Returns
    -------
    lstr : string
        Lambda function of the given string.

    Examples
    --------
    >>> x = np.array([[1, 3], [2, 2], [3, 1], [4, 7]]).T
    >>> c = np.array([5])
    >>> expr = 'x[0]*x[1]+c[0]'
    >>> f = common_functions.get_function(expr, const=True)
    >>> f(c, x)
    [ 8  9  8 33]
    """

    lstr = get_lambda_string(expr, const=const)

    return eval(lstr)


# ------------------------------------------------------------ #
#                       Fake sets
# ------------------------------------------------------------ #
# If we use sets (for primitive and terminal sets), then we need
# to convert them to list every time we want to use np.random.choice
# function. This also causes differences with same seed because the
# list is a different order on some runs. Perhaps this could be
# fixed by seeding random too (non-numpy version.)


def union(list1, list2):
    """Union two lists."""

    new_list = copy.copy(list1)

    for l2 in list2:

        # only add element that are not already in list1
        if l2 not in new_list:

            new_list.append(l2)

    return new_list


def intersection(list1, list2):
    """Get every element contained in both lists."""

    new_list = []

    for l1 in list1:

        if l1 in list2:

            new_list.append(l1)

    return new_list


def difference(list1, list2):
    """Set minus. Get everything in list1 unless it is also in list2."""

    new_list = copy.copy(list1)

    for l2 in list2:

        if l2 in new_list:

            new_list.remove(l2)

    return new_list


# ------------------------------------------------------------ #
#                      Folder Setup
# ------------------------------------------------------------ #


def get_folder(parameters):
    """Given the values of necessary parameters,
    create the appropriate folder name.

    Parameters
    ----------
    parameters : dict or collections.OrderedDict
        Each key is the name (in short form)
        and each value is the value of that
        parameter. This could be boolean values.

    Returns
    -------
    folder_name : str
        A string that indicates the parameters
        that will be used in the experiment that
        will populate this folder. Thus, the str
        is appropriate for folders.
    """

    str_values = [str(int(parameters[key])) for key in parameters]

    return '_'.join(key+str_values[i] for i, key in enumerate(parameters))
