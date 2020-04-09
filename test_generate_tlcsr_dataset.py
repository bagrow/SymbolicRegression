from generate_tlcsr_dataset import generate_regression_dataset, generate_functions, get_tlcsr_dataset

import numpy as np

# new/updated unit test
def test_generate_regression_dataset():

	F = [lambda x: x[0]**2,
		 lambda x: x[0]+x[1]]

	As = [[-1],
		  [-1, 0]]
	Bs = [[1],
		  [0, 1]]
	num_points = 5

	for f, A, B in zip(F, As, Bs):

		output = generate_regression_dataset(f, A=A, B=B, num_points=num_points)

		assert output[1].shape[1] == 1, 'y has more than one column'
		assert output[0].shape[0] == output[1].shape[0], 'X, y do not have the same number of columns'

		y = f(output[0].T)

		assert np.all(y == output[1].flatten()), 'f(X) != y'


# new/updated-------- unit test
def test_generate_functions():

	primitive_set = list(primitive_set)
	terminal_set = list(terminal_set)

	output = generate_functions(shape, primitive_set, terminal_set)

# new/updated unit test
def test_get_tlcsr_dataset():

	F = [lambda x, c: c[0]*x[0],
		 lambda x, c: c[0]*x[0] + c[1],
		 lambda x, c: c[0]*x[0]**2 + c[2]*x[0] + c[1]]
	num_consts = [1, 2, 3]

	for f, nc in zip(F,num_consts):
		output = get_tlcsr_dataset(f=f, num_points=5, num_datasets=10, num_consts=nc)

		assert len(output) == 10, 'Not the correct number of dataset created'

		for i, dataset in enumerate(output):
			assert (len(dataset[0]) == 5) and (len(dataset[1]) == 5), 'Incorrect number of points created in dataset '+str(i)
