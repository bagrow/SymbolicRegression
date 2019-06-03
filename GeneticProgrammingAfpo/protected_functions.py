import GeneticProgrammingAfpo.pickling as pickling

# import numpy as np
# from interval import interval, inf

import os


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.pickle')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.pickle')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    functions = pickling.unpickle_this(pickle_path)

    for key in functions:
        exec(key + '= functions[key]')

# if pickled_path does not exist
else:
    print('protected functions have not yet been pickled. Import protected_functions_writer once to pickle them.')
