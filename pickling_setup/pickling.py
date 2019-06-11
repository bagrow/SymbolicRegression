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
    pickled = False

    while not pickled:

        try:

            with open(save_loc, 'wb') as f:

                dill.dump(data, f)
                pickled = True

        except EOFError:

            pass


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

    # If multiple instances try to use this file
    # at the same time you will get EOFError.
    unpickled = False

    while not unpickled:

        try:

            with open(save_loc, 'rb') as f:

                data = dill.load(f)
                unpickled = True

        except EOFError:
            pass

    return data
