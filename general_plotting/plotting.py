import general_plotting.data_manipulation as dm

import matplotlib.pyplot as plt
import numpy as np

def get_emprical_cumulative_distribution_funtion(X):
    """Get the probability that x_i > X after X has
    been sorted (that is x_i is >= i+1 other x's).

    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF

    Returns
    -------
    p : list
        A list of the same length as X that
        give the probability that x_i > X
        where X is a randomly selected value
        and i is the index.
    """

    X_sorted = sorted(X)

    n = len(X)

    p = [i/n for i, x in enumerate(X)]

    return p, X_sorted


def plot_emprical_cumulative_distribution_funtion(X, labels=True, label=None, color=None):
    """Use get_emprical_cumulative_distribution_funtion to plot the CDF.

    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF
    labels : bool (default=True)
        If true, label x-axis x and y-axis Pr(X < x)
    label : str (default=None)
        The legend label.
    color : str (default=None)
        Color to used in plot. If none, it will not
        be pasted to plt.step.
    """

    p, X = get_emprical_cumulative_distribution_funtion(X)

    if color is None:
        plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4, label=label)

    else:
        plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4, label=label, color=color)

    if labels:

        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')


def plot_confidence_interval(x_data, y_data, color='C0', label=None):
    """Plot the mean of the columns of x_data and y_data and the
    95% confidence interval.

    Parameters
    ----------
    x_data : 2D list
        The x_data, which may not be exactly consistent.
    y_data : 2D list
        The y_data which will be averaged at each x-value.
    color : str (default='C0')
        The color to use when plotting.
    lable : str
        The legend label.

    Returns
    -------
    x : 1D np.array
        The mean of the columns of x_data after it
        was sent to organize_data.
    y : 1D np.array
        The mean of the columns of y_data after it
        was sent to organize_data.
    """

    _x_data, _y_data = dm.organize_data(x_data, y_data)
    print('done organize_data')
    # Get the average of the columns
    x = np.mean(_x_data, axis=0)
    y = np.mean(_y_data, axis=0)

    # Get the confidence interval
    y_upper = np.percentile(_y_data, 97.5, axis=0)
    y_lower = np.percentile(_y_data, 2.5, axis=0)

    # plot
    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y_lower, y_upper, alpha=0.5, facecolor=color, edgecolor='none')
    print('done plot_confidence_interval')
    return x, y


def plot_standard_error(x_data, y_data, color='C0', label=None):
    """Plot the mean of the columns of x_data and y_data and the
    standard error.

    Parameters
    ----------
    x_data : 2D list
        The x_data, which may not be exactly consistent.
    y_data : 2D list
        The y_data which will be averaged at each x-value.
    color : str (default='C0')
        The color to use when plotting.
    lable : str
        The legend label.

    Returns
    -------
    x : 1D np.array
        The mean of the columns of x_data after it
        was sent to organize_data.
    y : 1D np.array
        The mean of the columns of y_data after it
        was sent to organize_data.
    """

    _x_data, _y_data = dm.organize_data(x_data, y_data)
    print('done organize_data')
    # Get the average of the columns
    x = np.mean(_x_data, axis=0)
    y = np.mean(_y_data, axis=0)

    std = np.std(_y_data, axis=0)

    # Get the confidence interval
    y_upper = y + std/np.sqrt(_y_data.shape[0])
    y_lower = y - std/np.sqrt(_y_data.shape[0])

    # plot
    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y_lower, y_upper, alpha=0.5, facecolor=color, edgecolor='none')
    print('done plot_confidence_interval')
    return x, y


def plot_average(x_data, y_data, *args, **kwargs):
    """Plot the mean of the columns of x_data and y_data.

    Parameters
    ----------
    x_data : 2D list
        The x_data, which may not be exactly consistent.
    y_data : 2D list
        The y_data which will be averaged at each x-value.
    color : str (default='C0')
        The color to use when plotting.
    lable : str
        The legend label.

    Returns
    -------
    x : 1D np.array
        The mean of the columns of x_data after it
        was sent to organize_data.
    y : 1D np.array
        The mean of the columns of y_data after it
        was sent to organize_data.
    """

    _x_data, _y_data = dm.organize_data(x_data, y_data)

    # Get the average of the columns
    x = np.mean(_x_data, axis=0)
    y = np.mean(_y_data, axis=0)

    # plot
    plt.plot(x, y, *args, **kwargs)

    return x, y

def figure_ylabel(fig, ylabel):
    """Put a single ylabel for the figure
    even if using subplots.
    
    Parameters
    ----------
    fig : plt.figure()
        The figure to do this to.
    ylabel : str
        The y-axis label.
    """

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)
