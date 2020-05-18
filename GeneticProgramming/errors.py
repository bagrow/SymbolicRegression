import numpy as np

def RMSE(x, y, f):
    """RMSE = root mean squared error

    Parameters
    ----------
    x : np.array
        The input data where each column is a different
        variable and each row is for a different observation
    y : np.array
        The output data where there is only one column and
        just as many rows as x
    f : function
        The function that will produce y = f(x) if error = 0

    Returns
    -------
    error : float
        The square root of the sum of squares between y and f(x).
    """

    assert len(x) ==  len(y), 'The number of rows of x,y must be the same. x.shape='+str(x.shape)+', y.shape='+str(y.shape)

    return np.sqrt(np.mean(np.power(f(x.T) - y.flatten(), 2)))


def RSE(x, y, f):
    """RSE = relative squared error

    Parameters
    ----------
    x : np.array
        The input data where each column is a different
        variable and each row is for a different observation
    y : np.array
        The output data where there is only one column and
        just as many rows as x
    f : function
        The function that will produce y = f(x) if error = 0

    Returns
    -------
    error : float
        The sum of squares between y and f(x) divided by the sum of
        between mean(y) and y. Thus, if f is the average of y then
        RSE 1. If the prediction is better the RSE is smaller.
    """

    assert len(x) ==  len(y), 'The number of rows of x,y must be the same. x.shape='+str(x.shape)+', y.shape='+str(y.shape)

    return np.sum(np.power(f(x.T) - y.flatten(), 2))/np.sum(np.power(np.mean(y) - y.flatten(), 2))
