import numpy as np
from interval import interval, inf

import dill
import os


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


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.dill')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    functions = unpickle_this(pickle_path)

    for f in functions:
        exec(f)

else:

    print(pickle_path, 'does not exist.')
    print('Need to import GeneticProgrammingAfpo.protected_functions_writer to create it.')
