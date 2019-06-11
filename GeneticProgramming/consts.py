import numpy as np

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


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts_backup.dill')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    (population_size,
     max_generations,
     function_dict,
     required_children,
     math_translate,
     math_translate_interval_arithmetic,
     simplification_rules) = unpickle_this(pickle_path)

    functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]

else:

    print(pickle_path, 'does not exist.')
    print('Need to import GeneticProgrammingAfpo.consts_writer to create it.')
