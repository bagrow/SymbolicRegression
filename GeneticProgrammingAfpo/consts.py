import GeneticProgrammingAfpo.pickling as pickling

import numpy as np

import os

pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts_backup.dill')
print('in consts')

# change stuff from here if it exists
if os.path.isfile(pickle_path):

    (population_size,
     max_generations,
     function_dict,
     required_children,
     math_translate,
     math_translate_interval_arithmetic,
     simplification_rules) = pickling.unpickle_this(pickle_path)

    functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]

else:

    print(pickle_path, 'does not exist.')
    print('Need to import GeneticProgrammingAfpo.consts_writer to create it.')
