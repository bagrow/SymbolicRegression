import matplotlib

matplotlib.rcParams['font.size'] = 22

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import itertools

def get_emprical_cumulative_distribution_funtion(X):
    """Get the probability that x_i > X after X has
    been sorted (that is x_i is >= i+1 other x's).

    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF

    Returns
    -------
    p : list
        A list of the same length as X that
        give the probability that x_i > X
        where X is a randomly selected value
        and i is the index.
    """

    X_sorted = sorted(X)
    print(X_sorted)

    n = len(X)

    p = [(i)/n for i, x in enumerate(X)]

    return p, X_sorted


def plot_emprical_cumulative_distribution_funtion(X, labels=True):

    p, X = get_emprical_cumulative_distribution_funtion(X)

    plt.step(X, p, 'o-', where='post', linewidth=0.5, ms=4)

    if labels:

        plt.yticks(np.linspace(0, 1, 6))
        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')


# get data for recurrent version
base_path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'five_hours', 'recurrent')

offset = 0
all_data_rec = []

for rep in range(offset, offset+10):

    path = os.path.join(base_path, 'best_ind_rep'+str(rep)+'.csv')
    
    data = pd.read_csv(path).iloc[:, 1].values
    data = data[~np.isnan(data)]

    all_data_rec.append(data)

all_data_rec = list(itertools.chain(*all_data_rec))

# get data for non recurrent version
base_path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'five_hours', 'no_recurrent')

offset = 10
all_data_no_rec = []

for rep in range(offset, offset+10):

    path = os.path.join(base_path, 'best_ind_rep'+str(rep)+'.csv')
    
    data = pd.read_csv(path).iloc[:, 1].values
    data = data[~np.isnan(data)]

    all_data_no_rec.append(data)

all_data_no_rec = list(itertools.chain(*all_data_no_rec))

plt.figure(figsize=(6.4*1.5, 4.8*1.5))
plot_emprical_cumulative_distribution_funtion(all_data_rec)
plot_emprical_cumulative_distribution_funtion(all_data_no_rec)
plt.legend(['Recurrent, Error Input', 'No Recurrent, Fitness Input'])
plt.xlabel('Test Fitness (x)')
# plt.ylabel('')
plt.show()

