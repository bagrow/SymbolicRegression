import GeneticProgrammingAfpo.pickling as pickling

import numpy as np
from interval import interval, inf

import os


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.dill')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    functions = pickling.unpickle_this(pickle_path)

    for f in functions:
        exec(f)

else:

    print(pickle_path, 'does not exist.')
    print('Need to import GeneticProgrammingAfpo.protected_functions_writer to create it.')
