import numpy as np

def evolutionary_strategy(f, train_dataset, val_dataset,
                          npop=50, noise_std=0.1, learning_rate=0.001,
                          num_iter=3000):
    """Natrual Evolutaionary startegy modified from
    https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

    Parameters
    ----------

    Returns
    -------
    best : tuple
        The best inputs found and their score.
    """


    # start the optimization
    w = np.random.randn(3)  # our initial guess is random

    best = (None, float('inf'))

    for i in range(num_iter):

        # print current fitness of the most likely parameter setting
        if i % 20 == 0:
            print('iter %d. w: %s, reward: %f' % (i, str(w), f(w, train_dataset)))

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, 3)    # samples from a normal distribution N(0,1)
        R = np.zeros(npop)

        for j in range(npop):

            w_try = w + learning_rate*N[j]  # jitter w using gaussian of sigma 0.1
            R[j] = f(w_try, train_dataset)     # evaluate the jittered version
            val_error = f(w_try, val_dataset)

            # save best
            if val_error < best[1]:
                best = (w_try, val_error)

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)

        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w - learning_rate/(npop*noise_std) * np.dot(N.T, A)

    return best


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


    best = evolutionary_strategy(f, train_data, val_data)
    print('best', best)
