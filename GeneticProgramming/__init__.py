import os

print('in __init__')

pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.dill')

if os.path.exists(pickle_path):

	from . import protected_functions
	from .protected_functions import *

	from . import consts
	from .consts import *

	# Import Classes
	from . import Tree
	from .Tree import Tree

	from . import Individual
	from .Individual import Individual

	from . import GeneticProgramming
	from .GeneticProgramming import GeneticProgramming

	from . import GeneticProgrammingAfpo
	from .GeneticProgrammingAfpo import GeneticProgrammingAfpo

	# Import Functions/Global Variables
	from .common_functions import *
	from .data_setup import *
