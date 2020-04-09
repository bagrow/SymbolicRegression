import create_dataset_from_function as cdff

import numpy as np

def check_output(ans, out, msg='Failure'):
	assert np.all(ans == out), msg


def test_get_y():
    X = [np.array([0, 1, 2])[:, None],
         np.array([[0, 1, 2], [3, 4, 5]]).T]
    F = [lambda x: x[0] + 1,
         lambda x: x[0] + x[1]]
    answers = [np.array([1, 2, 3])[:, None], # f(x) = x[0] + 1
               np.array([3, 5, 7])[:, None]   # f(x) = x[0] + x[1]
              ]

    for x, f, ans in zip(X, F, answers):
        dataset = cdff.get_y(x, f)
        yield check_output, ans, dataset


def test_combine_x_y():

    X = [np.array([0, 1, 2])[:, None],
         np.array([[0, 1, 2], [3, 4, 5]]).T]
    Y = [np.array([1, 2, 3])[:, None], # f(x) = x[0] + 1
         np.array([3, 5, 7])[:, None]   # f(x) = x[0] + x[1]
        ]
    answers = [[[1., 0.],
                [2., 1.],
                [3., 2.]],

               [[3., 0., 3.],
                [5., 1., 4.],
                [7., 2., 5.]]]

    for x, y, ans in zip(X, Y, answers):
        dataset = cdff.combine_x_y(x, y)
        yield check_output, ans, dataset


def test_get_dataset():

	X = [np.array([0, 1, 2])[:, None]]
	F = [lambda x: x[0] + 1]
	answers = [[[1., 0.],
				[2., 1.],
				[3., 2.]]]

	for x, f, ans in zip(X, F, answers):
		dataset = cdff.get_dataset(x, f)
		yield check_output, ans, dataset


