import numpy as np

def combine_x_y(x, y):
	"""
	Create a dataset by combining the inputs (x)
	and the outputs (y).

	Paramters
	---------
	x : np.array
		Each column is a variable and each row
		is an observation.
	y : np.array
		Single column array containing the outputs.

	Returns
	-------
	datasets : np.array
		The 0-th column is y and the remaining columns are
		x[0], x[1], ...

	Examples
	--------
	>>> x = np.array([0, 1, 2])[:, None]
	>>> y = np.array([1, 2, 3])[:, None]
	>>> combine_x_y(x, y)
	[[1 0]
	 [2 1]
	 [3 2]]
	"""

	dataset = np.hstack((y,x))
	
	return dataset


def get_y(x, f):
	"""Given the inputs (x) for the function (f),
	get the outputs (y) as a column array.

	Parameters
	----------
	x : np.array
		Each column is a variable and each row
		is an observation.
	f : lambda function
		The inputs to this function is a row x.
		The only input variable is x and the specific
		variables are references with indices.

	Returns
	-------
	y : np.array
		Single column array containing the outputs.

	Examples
	--------
	>>> x = np.array([0, 1, 2])[:, None]
	>>> f = lambda x: x[0] + 1
	>>> get_y(x, f)
	[[1]
	 [2]
	 [3]]
	"""

	y = f(x.T)[:, None]
	return y

def get_dataset(x, f):
	"""
	Create a dataset given the inputs (x)
	and the underlying target function f.

	Paramters
	---------
	x : np.array
		Each column is a variable and each row
		is an observation.
	f : lambda function
		The inputs to this function is a row x.
		The only input variable is x and the specific
		variables are references with indices.

	Returns
	-------
	datasets : np.array
		The 0-th column is f(x) and the remaining columns are
		x[0], x[1], ...

	Example
	-------
	>>> x = np.array([0, 1, 2])[:, None]
	>>> f = lambda x: x[0] + 1
	>>> get_dataset(x, f)
	[[1 0]
	 [2 1]
	 [3 2]]
	"""

	y = get_y(x, f)
	dataset = combine_x_y(x, y)
	
	return dataset


if __name__ == '__main__':

	x = np.array([0, 1, 2])[:, None]
	f = lambda x: x[0] + 1
	y = get_y(x, f)
	print(y)

	x = np.array([0, 1, 2])[:, None]
	f = lambda x: x[0] + 1
	dataset = get_dataset(x, f)
	print(dataset)

	x = np.array([0, 1, 2])[:, None]
	y = np.array([1, 2, 3])[:, None]
	dataset = combine_x_y(x, y)
	print(dataset)