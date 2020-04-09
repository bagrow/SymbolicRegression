import create_dataset_from_function as cdff
import numpy as np

import itertools

# new/updated function
def generate_regression_dataset(f, noise_std=0, num_points=[20],
								use_grid=True, A=[-1], B=[1]):
	"""Generate a regression dataset
	(x, y) data for some underlying function.

	Parameters
	----------
	f : function
		The underlying function: f(x) = y
	noise_std : float (default=0)
		The stardard deviation of the noise
	num_points : int or list of ints (default=[20])
		If int, the number of rows (observation) desired
		in your dataset. Otherwise, the number of points
		in each xi direction.
	use_grid : bool (default=True)
		If true, uniformly space the points. Otherwise,
		randomly place them in [a,b].
	A : list of floats (default=[-1])
		The left end point(s) of input variables
	B : list of floats (default=[1])
		The right end point(s) of input varaibles

	Returns
	-------
	X : np.array
		Inputs to underlying function f. The columns
		are inputs (x0, ..., xn). There is one row for every
		observation in dataset (X, y).
	y : np.array
		Output of the underlying function given X. This array
		is 2D and contains only one column. There is one row for every
		observation in dataset (X, y).
	"""

	assert len(A) == len(B), 'Number of endpoints do not match'

	if type(num_points) is not int:
		assert len(A) == len(num_points), 'Number of endpoints do not match'

	num_inputs = len(A)

	if type(num_points) is int:
		x = [np.linspace(a, b, num_points) for a,b in zip(A, B)]
	else:
		x = [np.linspace(a, b, n) for a,b,n in zip(A, B, num_points)]

	X = np.array(list(itertools.product(*x)))
	y = f(X.T)[:, None] + np.random.normal(0, noise_std, (len(X), 1))
	print('y', y)

	return X, y


# new/updated --------function
def generate_functions(shape, primitive_set, terminal_set):
	"""Given a shape (tree), a primitive set, and a terminal
	set generate all functions possible.

	Parameters
	----------
	shape : Tree?
		The tree shape
	primitive_set : set/list
		All primitives that are allowed to be
		used in the tree. Note that the shape
		may make it impossible to use some of these
		element.
	terminal_set : set/list
		All terminals that are allowed to be used in
		the tree.

	Returns
	-------
	functions : list
		A list of functions of the same shape, which 
		is given by shape.
	"""

	primitive_set = list(primitive_set)
	terminal_set = list(terminal_set)

# new/updated function
def get_tlcsr_dataset(f=None, a=-1, b=1, cmin=-10, cmax=10, num_points=20, num_datasets=1, num_consts=1):
	"""Get many regression datasets
	from a single funtion by changing the constants

	Paramters
	---------
	f : function
		f(x,c) = y

	Returns
	-------
	datasets : list
	"""

	X = np.linspace(a, b, num_points)[:, None]

	datasets = []

	for _ in range(num_datasets):

		c = np.random.uniform(cmin, cmax, num_consts)

		y = f(X.T, c)
		print(y)
		datasets.append((X,y))

	return datasets
