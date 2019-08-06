import pickling_setup.pickling as pickling
import numpy as np

import os
import dill


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_protected_functions_backup.dill')

if os.path.isfile(pickle_path_backup):

    functions = pickling.unpickle_this(pickle_path_backup)

    for f in functions:
        exec(f)

    pickle_path_exists = True

else:

    pickle_path_exists = False
    print(pickle_path_backup, 'does not exist.')
    print('Run anything with run_gp.py to create it.')
    exit()

# All the following functions allow
# single input function to pretend to
# be two input functions.

def sin2(x, y):
    return np.sin(x)


def cos2(x, y):
    return np.cos(x)


def psqrt2(x, y):
    return psqrt(x)


def abs2(x, y):
    return np.abs(x)


def id2(x, y):
    """This is the two variable version
    of the identity function"""

    return x


functions.append(dill.source.getsource(sin2))
functions.append(dill.source.getsource(cos2))
functions.append(dill.source.getsource(psqrt2))
functions.append(dill.source.getsource(abs2))
functions.append(dill.source.getsource(id2))

# Pickle them all
if not os.path.exists(os.path.dirname(pickle_path)):
    os.makedirs(os.path.dirname(pickle_path))

# Overwrite pickle_path
pickling.pickle_this(functions, pickle_path)
