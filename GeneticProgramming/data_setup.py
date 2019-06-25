import numpy as np
import pandas as pd

import os

# Functions in this file are used to generate training,
# validation, and testing data.


def get_all_data(X, f):
    """Given input data, compute output data using function f and put the
    data in a single list which is returned.

    Parameters
    ----------
    X : np.array
        Input data. This array must have the same number of columns as
        the function f takes as inputs.
    f : function
        A function that can take X as input

    Returns
    -------
    data : np.array
        The first column is the output of f(X) and the other columns
        are the columns of X. The data is sorted by first column
        of X.

    See Also
    --------
    create_input_data

    Examples
    --------
    >>> x = data_setup.get_input_data(np.random.RandomState(0), [0, 1], [1,3], data_size=3)
    >>> x
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,
            0.64589411],
           [1.87517442, 2.783546  , 2.92732552, 1.76688304, 2.58345008,
            2.05778984]])
    >>> f = lambda x: x[1]**3 * x[0]**2
    >>> dataset = data_setup.get_all_data(x, f)
    >>> dataset
    array([[ 3.09474223,  0.4236548 ,  2.58345008],
           [ 1.6376844 ,  0.54488318,  1.76688304],
           [ 1.98597872,  0.5488135 ,  1.87517442],
           [ 9.1139527 ,  0.60276338,  2.92732552],
           [ 3.63517822,  0.64589411,  2.05778984],
           [11.03156952,  0.71518937,  2.783546  ]])

    """

    data = np.vstack((f(X), X)).T
    data = np.array(sorted(data, key=lambda x: x[1]))

    return data


def get_input_data(rng, A, B, data_size=20, random_spacing=True):
    """Create input data in the interval(s) [a1, b1], [a2, b2], etc.

    Paremters
    ---------
    rng : random number generator
        (ex. np.random.RandomState(0))
    A : iterable
        This specifies all the left endpoints of interval from
        which data will be selected.
    B : iterable
        The values specify the right endpoints of the interval
        from which data will be selected.
    data_size : int, optional
        The number of data points created will be len(A)*data_size
    random_spacing : bool, optional
        If false, select data points with uniform spacing.

    Returns
    -------
    x_data : np.array
        The generated data. Where the zeroth row is between
        A[0] and B[0] and the first row is between A[1], B[1],
        and so on.

    Examples
    --------
    >>> x = ds.get_input_data(np.random.RandomState(0), [0, 1], [1,3])
    >>> x
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,
            0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152,
            0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606,
            0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215,
            0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443,
            0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
            0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898 ,
            0.6176355 , 0.61209572, 0.616934  , 0.94374808, 0.6818203 ],
           [1.7190158 , 1.87406391, 2.39526239, 1.12045094, 2.33353343,
            2.34127574, 1.42076512, 1.2578526 , 1.6308567 , 1.72742154,
            2.14039354, 1.87720303, 2.97674768, 1.20408962, 1.41775351,
            1.32261904, 2.30621665, 1.50658321, 1.93262155, 1.48885118,
            1.31793917, 1.22075028, 2.31265918, 1.2763659 , 1.39316472,
            1.73745034, 2.64198646, 1.19420255, 2.67588981, 1.19219682,
            2.95291893, 1.9373024 , 2.95352218, 2.20969104, 2.47852716,
            1.07837558, 1.56561393, 1.24039312, 1.5922804 , 1.23745544]])
    >>> x.shape
    (2, 40)
    """

    if random_spacing:

        if type(A) == int or type(A) == float:

            x_data = A + (B - A) * rng.rand(data_size)

        else:

            x_data = np.array(
                [a + (b - a) * rng.rand(data_size * len(A)) for a, b in zip(A, B)])
    else:

        if type(A) == int or type(A) == float:

            step = (B - A) / (data_size * len(A) - 1)
            x_data = np.arange(start=A, stop=B + step, step=step)

        else:

            x_data = []

            for a, b in zip(A, B):

                step = (b - a) / (data_size - 1)
                x_data.append(np.arange(start=a, stop=b + step, step=step))

            x_data = np.array(x_data)

    return x_data


def save_data(filename, data):
    """Save the data to a csv file.

    Parameters
    ----------
    filename : string
        File in which data will be saved.
    data : np.array
        Data to be saved.
    """

    df = pd.DataFrame(data)
    cols = ['y'] + ['x' + str(i) for i in range(data.shape[1] - 1)]
    df.to_csv(filename, header=cols, index=False)


def get_datasets(rng, f, A, B, filename=None, noise_percent=None,
                 noise_std=0, data_size=20, random_spacing=True,
                 num_datasets=2):
    """Use the function f to generate input value in [a1, b1], ... and outputs.
    Use range of training data to determine the amount of Gaussian noise to add.
    Get two version of data: training and validation data.

    Parameters
    ----------
    rng : random number generator
        (ex. np.random.RandomState(0))
    A : iterable
        This specifies all the left endpoints of interval from
        which data will be selected.
    B : iterable
        The values specify the right endpoints of the interval
        from which data will be selected.
    data_size : int, optional
        The number of data points created will be len(A)*data_size
    noise_percent : float in [0, 100] (optional)
        This parameter is only effective if noise_std is None.
        This will set noise_std to
        noise_percent * (length of range of function).
    noise_std : float (optional)
        This number will be used as the standard deviation
        of the normal distribution from which the amount of
        noise is chosen.
    filename : string or None (optional)
        If None (default), data is not saved.
    random_spacing : bool (optional)
        If false, select data points with uniform spacing.
    num_datasets :  int (optional)
        Can specify the number of times to generate
        data. Default is 2 for training and validation
        since testing dataset will probably have a different
        number of data points.

    Returns
    -------
    data : np.array of shape (2, len(X), len(X[0])+1)
        The 2 is for training data (index=0) and validation
        data (index=1). For the subarrays, the first column
        is the output of f(X) and the other columns
        are the columns of X. The data is sorted by first
        column of X.

    Examples
    --------
    >>> f = lambda x: x**2
    >>> data = data_setup.get_datasets(np.random.RandomState(0), f, [0], [1], noise_std=0.01, data_size=5)
    >>> data
    [[[0.18092382 0.4236548 ]
      [0.31144042 0.54488318]
      [0.30880664 0.5488135 ]
      [0.36454044 0.60276338]
      [0.51593446 0.71518937]]

     [[0.15036414 0.38344152]
      [0.20642336 0.43758721]
      [0.41512762 0.64589411]
      [0.79838976 0.891773  ]
      [0.92010496 0.96366276]]]
    >>> data.shape
    (2, 5, 2)
    """

    # get training data input
    r_data = get_input_data(rng, A, B, data_size=data_size,
                            random_spacing=random_spacing)

    # get validation data input
    v_data = get_input_data(rng, A, B, data_size=data_size,
                            random_spacing=random_spacing)

    # get the outputs
    r_data = get_all_data(r_data, f)
    v_data = get_all_data(v_data, f)

    if noise_std is None:

        yr_data = [p[0] for p in r_data]

        range_of_training = max(yr_data) - min(yr_data)

        noise_std = range_of_training * noise_percent / 100

    r_data[:, 0] += rng.normal(0, noise_std, size=len(r_data))
    v_data[:, 0] += rng.normal(0, noise_std, size=len(r_data))

    if filename is not None:

        # make directory if necessary
        directory = os.path.dirname(filename)

        if not os.path.exists(directory):

            os.makedirs(directory)

        # save the std for the noise in case it is forgotten
        with open(filename + '_std_dev.txt', 'w') as file:

            file.write(str(noise_std))

        save_data(filename + '_training.csv', r_data)
        save_data(filename + '_validation.csv', v_data)

    return np.array((r_data, v_data))


def get_extrapolation_and_interpolation_data(rng, f, A, B, noise_std=0,
                                             filename=None,
                                             data_size=20,
                                             random_spacing=True):
    """Create testing data in two categories: interpolation and extrapolation.
    Use the function f to generate input value in [a1, b1], ... and outputs.
    Use noise_std as the standard deviation of Gaussian noise to add.
    Get two version of data: interpolation (same domain as training/validation)
     and extrapolation (outside of domain of training/validation data) data.

    Parameters
    ----------
    rng : random number generator
        (ex. np.random.RandomState(0))
    A : iterable
        This specifies all the left endpoints of interval from
        which data will be selected.
    B : iterable
        The values specify the right endpoints of the interval
        from which data will be selected.
    data_size : int, optional
        The number of data points created will be len(A)*data_size
    noise_std : float (optional)
        This number will be used as the standard deviation
        of the normal distribution from which the amount of
        noise is chosen.
    filename : string or None (optional)
        If None (default), data is not saved.
    random_spacing : bool, optional
        If false, select data points with uniform spacing.

    Returns
    -------
    data : np.array of shape (2, len(X), len(X[0])+1)
        The 2 is for iterpolation data (index=0) and extrapolation
        data (index=1). For the subarrays, the first column
        is the output of f(X) and the other columns
        are the columns of X. The data is sorted by first
        column of X.

    Examples
    --------
    >>> f = lambda x: x**2
    >>> data = data_setup.get_extrapolation_and_interpolation_data(np.random.RandomState(0), f, [0], [1], noise_std=0.01, data_size=5)
    >>> data
    [[[ 0.18432651  0.4236548 ]
      [ 0.29508186  0.54488318]
      [ 0.29745155  0.5488135 ]
      [ 0.35371614  0.60276338]
      [ 0.51183022  0.71518937]]

     [[ 0.13390377 -0.35792788]
      [ 1.18796611  1.08345008]
      [ 1.65024232  1.283546  ]
      [ 1.82949254  1.35119328]
      [ 2.04406382  1.42732552]]]
    >>> data.shape
    (2, 5, 2)
    """

    i_data = get_input_data(rng, A, B, data_size=data_size,
                            random_spacing=random_spacing)

    # The interval [e_start, e_start+e_length] is centered
    # at the center of [a,b] and has twice the length
    e_length = np.array([2 * (b - a) for a, b in zip(A, B)])
    e_start = np.array([a - (b - a) / 2. for a, b in zip(A, B)])

    e_data = []

    while len(e_data) < data_size * len(A):

        entry = e_start + e_length * rng.rand(len(A))

        if np.all(A <= entry) and np.all(entry <= B):

            continue

        else:

            e_data.append(entry)

    ti_data = get_all_data(i_data, f)
    te_data = get_all_data(np.array(e_data).T, f)

    for i in range(len(ti_data)):

        ti_data[i, 0] += rng.normal(0, noise_std)
        te_data[i, 0] += rng.normal(0, noise_std)

    if filename is not None:

        save_data(filename + '_testing_interpolation.csv', ti_data)
        save_data(filename + '_testing_extrapolation.csv', te_data)

    return np.array((ti_data, te_data))


def read_data(filename, one_file=False):
    """Get training, validation, and testing
    (both interpolation and extrapolation) data.

    Parameters
    ----------
    filename : string
        Base location of data excluding _training.csv for example.
    one_file : bool (default=False)
        If True, only get the the one file. Also, nothing
        will be appended to th end of filename.

    Returns
    -------
    np.array
    The contents of the file specified or the
    contents of several files if one_file=False.
    """

    if one_file:
        df = pd.read_csv(filename)

        return df.iloc[:, :].values

    else:
        df = pd.read_csv(filename + '_training.csv')
        df2 = pd.read_csv(filename + '_validation.csv')
        df3 = pd.read_csv(filename + '_testing_interpolation.csv')
        df4 = pd.read_csv(filename + '_testing_extrapolation.csv')

        training = df.iloc[:, :].values
        validation = df2.iloc[:, :].values
        testing_int = df3.iloc[:, :].values
        testing_ext = df4.iloc[:, :].values

        return np.array([training,
                         validation,
                         testing_int,
                         testing_ext])


def split_data(rng, data, ratio, with_replacement=False):
    """Split the dataset by the given
    ratio that determines the relative
    sizes.

    Parmeters
    ---------
    rng : random number generator
        (ex. np.random.RandomState(0))
    data : 2d iterable
        The data to be split. Each row of data
        show be a complete datum.
    ratio : 1d iterable
        The ratio used in splitting. For example,
        use 1:2 to split data into two subsets:
        the first of size have of the second.
    with_replacement : bool (default=False)
        If True, the subsets of data may have
        elements in common.

    Examples
    --------
    """

    num_samples = len(data)
    sizes = np.multiply(num_samples/sum(ratio), ratio).astype(int, copy=False)

    if sum(sizes) != num_samples:
        sizes[-1] += num_samples - sum(sizes)

    remaining_indices = range(len(data))
    dataset = []

    for s in sizes:

        indices = rng.choice(remaining_indices, size=s, replace=with_replacement)
        data_split = data[np.array(indices), :]
        dataset.append(data_split)

        remaining_indices = [i for i in range(len(data)) if i not in indices]

    return np.array(dataset[:2]), np.array(dataset[2])
