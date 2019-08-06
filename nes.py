import numpy as np

import time

def evolutionary_strategy(f, w, seed=0, npop=50, noise_std=0.1, learning_rate=0.001,
                          timeout=float('inf'), max_evals=float('inf'), args=()):
    """Natrual Evolutaionary startegy modified from
    https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

    Parameters
    ----------
    f : function
        The function to be minimized.
    w : list
        The inital weights.
    seed : int (default=0)
        The seed for the random
        number generator.
    npop : int (default=50)
        The size of the population.
    noise_std : float (default=0.1)
        The standard deviation when
        applying noise to existing solutions.
    learning_rate : float (default=0.001)
        The learning rate.
    timeout : float (default=float('inf'))
        The max amount of time allowed for
        computation.
    max_evals : float (default=float('inf'))
        The maximum number of evalutions
        of f (for training only).
        If reached, the algorithm
        will terminate.
    args : tuple
        Constant arguments to pass to f.

    Returns
    -------
    best : tuple
        The best inputs found and their score.
    """

    assert timeout != float('inf') or max_evals != float('inf'), ('Loop will never end'
                                                                  'timeout and max_evals'
                                                                  'are both infinity.')

    start_time = time.time()

    rng = np.random.RandomState(seed)
    # print(num_weights)

    # start the optimization
    # w = rng.randn(num_weights)  # our initial guess is random

    # best = (error, weights)
    # best = (float('inf'), None)
    total_evals = 0

    # for i in range(num_iter):
    while True:

        elapsed_time = time.time() - start_time

        # Taking too long?
        if elapsed_time > timeout:
            break

        # Too many evaluations?
        if total_evals > max_evals:
            break

        # print current fitness of the most likely parameter setting
        if total_evals//npop % 20 == 0:
            print('reward: %f, hours: %f' % (f(w, *args), elapsed_time/3600.))

        # initialize memory for a population of w's, and their rewards
        N = rng.randn(npop, len(w))    # samples from a normal distribution N(0,1)
        R = np.zeros(npop)

        for j in range(npop):

            w_try = w + learning_rate*N[j]  # jitter w using gaussian
            R[j] = f(w_try, *args)     # evaluate the jittered version
            # val_error = f(w_try, val_dataset)

            # # save best
            # if val_error < best[0]:
            #     best = (val_error, w_try)

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)

        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w - learning_rate/(npop*noise_std) * np.dot(N.T, A)

        total_evals += npop

    return w


if __name__ == '__main__':

    np.random.seed(0)

    solution = np.array([2.5, 0.1, -0.3])

    target = lambda x: np.dot(solution, [x**2, x, np.ones_like(x)])

    x = np.linspace(-1, 1, 100)

    dataset = np.vstack((target(x), x)).T

    val_split = 0.2

    val_indices = np.random.choice(len(x), replace=False, size=int(len(x)*val_split))
    train_indices = np.array([i for i in range(len(x)) if i not in val_indices])

    train_data = dataset[train_indices, :]
    val_data = dataset[val_indices, :]

    f_hat = lambda w, x: np.dot(w, [x**2, x, np.ones_like(x)])

    # the function we want to optimize
    def f(w, dataset):
        # here we would normally:
        # ... 1) create a neural network with weights w
        # ... 2) run the neural network on the environment for some time
        # ... 3) sum up and return the total reward

        # but for the purposes of an example, lets try to minimize
        # the L2 distance to a specific solution vector. So the highest reward
        # we can achieve is 0, when the vector w is exactly equal to solution
        reward = np.sqrt(np.sum(np.square(f_hat(w, dataset[:, 1])-dataset[:, 0])))

        return reward


    best = evolutionary_strategy(f, num_weights=3, args=(train_data,), timeout=3)
    print('best', best)
