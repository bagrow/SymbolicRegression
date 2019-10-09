import numpy as np
import pandas as pd
import scipy.stats

import os

np.random.seed(0)

gp_data = np.random.uniform(0, 0.05, size=100)

# get equation adjuster (vertical shift) data
path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment2')

ea_data = []

for rep in range(100):

    data = pd.read_csv(os.path.join(path, 'best_ind_rep'+str(rep)+'.csv'))['test error on test function'].values
    data = data[~np.isnan(data)]
    
    ea_data.append(data)

ea_data = np.array(ea_data).flatten()

print(ea_data)

result = scipy.stats.wilcoxon(ea_data, gp_data, alternative='less')
print(result)

# result = scipy.stats.mannwhitneyu(ea_data, gp_data, alternative='less')
# print(result)
