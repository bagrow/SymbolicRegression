import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

import os

target_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14', 'keijzer15', 'r1', 'r2', 'r3']

base_path = os.path.join(os.environ['GP_DATA'], 'gp_sgp_comp')

# get gp data
gp_base_path = os.path.join(base_path, 'gp')
sgp_base_path = os.path.join(base_path, 'semantic_gp')

gp_test_error = {}
sgp_test_error = {}

fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)

axes = axes.flatten()

for i, name in enumerate(target_names):

    plt.sca(axes[i])

    gp_test_error[name] = []
    sgp_test_error[name] = []

    for rep in range(100):

        file = os.path.join(gp_base_path, name, 'best_data_rep'+str(rep)+'.csv')
        sgp_file = os.path.join(sgp_base_path, 'FPM_0.0_1.0_'+name+'_'+str(rep)+'.log')

        test_data = pd.read_csv(file)['Testing Error'].values
        sgp_test_data = pd.read_csv(sgp_file)['min_fitness'].values

        gp_test_error[name].append(test_data)
        sgp_test_error[name].append(sgp_test_data)

    plt.semilogy(np.median(gp_test_error[name], axis=0), color='C0')
    plt.semilogy(np.median(sgp_test_error[name], axis=0), color='C1')
    plt.title(name)

    if i in (9, 10):
        plt.xlabel('Generations')

    if i == 4:
        plt.ylabel('Median Test Error of Best Individuals')

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'figures', 'plot_gp_vs_semantic_gp.pdf'))

# get p-value
table = []

for name in target_names:

    gp_final_best = [x[-1] for x in gp_test_error[name]]
    sgp_final_best = [x[-1] for x in sgp_test_error[name]]

    _, pvalue = scipy.stats.mannwhitneyu(gp_final_best, sgp_final_best, alternative='greater')

    table.append([round(np.median(gp_final_best),4), round(np.median(sgp_final_best),4), '{:0.3e}'.format(pvalue)])

df = pd.DataFrame(table, columns=['Median Test Error in GP', 'Median Test Error in Semantic GP', 'p-value'], index=target_names)
df.to_csv(os.path.join(base_path, 'figures', 'stat_plot_gp_vs_semantic_gp.csv'))
