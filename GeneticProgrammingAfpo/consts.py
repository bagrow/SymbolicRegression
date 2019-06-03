from GeneticProgrammingAfpo.protected_functions import *
import GeneticProgrammingAfpo.pickling as pickling

import numpy as np

import os

pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.pickle')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts_backup.pickle')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    (population_size,
     max_generations,
     functions,
     function_dict,
     required_children,
     math_translate,
     math_translate_interval_arithmetic,
     simplification_rules) = pickling.unpickle_this(pickle_path)

    for key in functions:
        exec('key = functions[key]')

# if pickled_path does not exist
else:

    print('Consts have not yet been pickled. Import consts_writer once to pickle them.')
    exit()

functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]
