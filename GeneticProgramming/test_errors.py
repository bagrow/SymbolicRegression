from errors import *

import numpy as np


def test_RMSE():

    X = [np.linspace(-1, 1, 20)[:, None],
         np.array([[-1],
                   [0],
                   [1]])]
    Y = [np.random.uniform(0, 10, 20)[:, None],
         np.array([[5],
                   [6],
                   [7]])]

    F = [lambda x: 0*x[0],
         lambda x: x[0]]

    answers = [np.sqrt(np.mean(Y[0]**2)),
               6]

    for x,y,f,ans in zip(X,Y,F,answers):
        output = RMSE(x, y, f)

        assert ans == output, 'output='+str(output)


def test_RSE():

    X = [np.linspace(-1, 1, 20)[:, None],
         np.array([[-1],
                   [0],
                   [1]])]
    Y = [np.random.uniform(0, 10, 20)[:, None],
         np.array([[5],
                   [6],
                   [7]])]

    F = [lambda x: np.mean(y),
         lambda x: x[0]]

    answers = [1,
               54]

    for x,y,f,ans in zip(X,Y,F,answers):
        output = RSE(x, y, f)

        assert ans == output, 'output='+str(output)


if __name__ == '__main__':
    test_RSE()
    test_RMSE()