import dill

# ------------------------------------------------------------ #
#            Save Constants and Protected Functions
# ------------------------------------------------------------ #

def pickle_this(data, save_loc):
    """Pickle data in save location save_loc.

    Parameters
    ----------
    data : tuple
        A tuple of the data to save.
    save_loc : str
        The location where the pickled data
        will be saved.
    """

    with open(save_loc, 'wb') as f:
        dill.dump(data, f)


def unpickle_this(save_loc):
    """Unpickle data from save location save_loc.

    Parameters
    ----------
    save_loc : str
        The location where the pickled data
        will be saved.

    Returns
    -------
    data : tuple
        A tuple of the data that was unpickled.
    """
    with open(save_loc, 'rb') as f:
        data = dill.load(f)

    return data
