import pickling_setup.pickling as pickling

import os


pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.dill')
pickle_path_backup = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts_backup.dill')

if not os.path.isfile(pickle_path_backup):

    print('Either', pickle_path_backup, 'does not exist.')
    print('To create it, run run_gp.py')
    exit()

(population_size,
 max_generations,
 function_dict,
 required_children,
 math_translate,
 math_translate_interval_arithmetic,
 simplification_rules) = pickling.unpickle_this(pickle_path_backup)

population_size = 100
max_generations = 30000

# Dictionary explaining how many children (inputs)
# are needed for each function.
required_children['sin2'] = 2
required_children['cos2'] = 2
required_children['sqrt2'] = 2
required_children['abs2'] = 2
required_children['id2'] = 2
required_children['_p_'] = 2    # this is a fake primitive for overwriting lisp

functions_by_input = [[key for key, value in required_children.items() if value == i] for i in range(1, 3)]

# and use this translation for function creation.
math_translate['sin2'] = 'sin2'
math_translate['cos2'] = 'cos2'
math_translate['sqrt2'] = 'psqrt2'
math_translate['abs2'] = 'abs2'
math_translate['id2'] = 'id2'

print('Writing Constants from PrimitiveSetTransitions')

# Overwrite the settings stored
# in pickle_path
pickling.pickle_this((population_size,
                      max_generations,
                      function_dict,
                      required_children,
                      math_translate,
                      math_translate_interval_arithmetic,
                      simplification_rules), pickle_path)
