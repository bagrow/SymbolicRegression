import equation_correction as ec

import matplotlib.pyplot as plt
import numpy as np

import os

def plot_nn_behavior(rep):

    filename = '/Users/rgrindle/GP_DATA/equation_adjuster/best_ind_rep'+str(rep)+'.csv'

    w, hidden_weights, hidden_values = ec.equation_adjuster_from_file(filename)

    hidden_weights = hidden_weights.reshape((len(hidden_values), len(hidden_values)))

    # datasets = ec.get_data_for_equation_corrector(rng=np.random.RandomState(0), num_targets=1,
    #                                               num_base_function_per_target=1, depth=3):

    # for function_string, dataset in datasets:

        # ec.run_equation_corrector(function_string, dataset, w, hidden_values, hidden_weights, activation=np.tanh,
        #                           adjustment=1, constant=0, num_iterations=5)

    rng = np.random.RandomState(0)

    data = []

    for signed_error in list(rng.uniform(-5, 5, size=50)) + [0.]:

        index, hidden_value = ec.evalutate_corrector_neural_network(w, signed_error, hidden_values,
                                                                    hidden_weights, activation=np.tanh)

        data.append((signed_error, index))

    data = np.array(data)

    error = data[:, 0]
    index = data[:, 1]

    plt.plot(error, index, 'o')
    plt.yticks([0, 1, 2], ['Positive', 'Negative', 'Zero'])
    plt.xlabel('Signed Error (input)')
    # plt.ylabel('Index (output)')

    plt.tight_layout()
    plt.savefig(os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'figures', 'weights_rep'+str(rep)+'.pdf'))


if __name__ == '__main__':

    plot_nn_behavior(0)
