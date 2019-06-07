import os

print('in __init__')

pickle_path = os.path.join(os.environ['GP_DATA'], 'pickled', 'GeneticProgrammingAfpo_consts.dill')

if os.path.exists(pickle_path):

	from . import protected_functions
	from .protected_functions import *

	from . import consts
	from .consts import *

	from . import common_functions
	from .common_functions import *

	# Import Classes
	from . import Tree
	from .Tree import Tree

	from . import Individual
	from .Individual import Individual

	from . import GeneticProgramming
	from .GeneticProgramming import GeneticProgramming

	from . import GeneticProgrammingAfpo
	from .GeneticProgrammingAfpo import GeneticProgrammingAfpo
