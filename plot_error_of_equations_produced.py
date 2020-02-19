# from load_tlcsr_network import load_tlcsr_network
# from TlcsrNetwork import TlcsrNetwork

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

options = {'use_k-expressions': True,
           'head_length': 15}

primitive_set = ['*', '+', '-']
terminal_set = ['x0']

# if args.use_constants:
#     terminal_set.append('#f')
#     terminal_set.append('const_value')

timelimit = 100

function_order = ['Nguyen-4',
                  'Koza-1',
                  'Koza-2',
                  'Koza-3',
                  'Nguyen-1',
                  'Nguyen-3',
                  'Validation']

function_strs = {'Nguyen-4': 'x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Nguyen-4
                 'Koza-1': 'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
                 'Koza-2': 'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
                 'Koza-3': 'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
                 'Nguyen-1': 'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
                 'Nguyen-3': 'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]', # Nguyen-3
                 'Validation': 'x[0]**4 + x[0]'
                }

color_dict = {'Koza-1': 'C2',
              'Koza-2': 'C5',
              'Koza-3': 'C6',
              'Nguyen-1': 'C7',
              'Nguyen-3': 'C8',
              'Nguyen-4': 'C9',
              'Validation': 'C4'}

# get weights
exp = 25

filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'figures', 'data_equations_produced.csv')
df = pd.read_csv(filename)

fig, axes = plt.subplots(ncols=6, nrows=30,
                         sharey=True, sharex=True, figsize=(6,20))

# axes = axes.flatten()

for rep in range(30):

    for test_index in range(6):

        plt.sca(axes[rep, test_index])

        for f in function_order:

            all_rewrite_data = df.loc[(df['rep'] == rep) & (df['dataset'] == f) & (df['test_index'] == test_index)]
            plt.plot(all_rewrite_data['t'], all_rewrite_data['RMSE'], color=color_dict[f])

        if test_index == 0 and rep == 29:
            plt.ylabel('RMSE at time $t+1$')
            plt.xlabel('$t$')

        if rep < 1:
            plt.title(function_order[test_index])

        plt.ylim([-.05, 2])

        print('.', flush=True, end='')

labels = color_dict.keys()
colors = color_dict.values()
custom_lines = [plt.Line2D([0], [0], color=c) for c in colors]
fig.legend(custom_lines, labels,
           ncol=4, handlelength=1,
           handletextpad=0.4, labelspacing=0.25,
           columnspacing=1.3,
           loc='center', bbox_to_anchor=(0.5, 0.99),
           fontsize=8)

plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.03, wspace=0.15, hspace=0.2)

filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'figures', 'plot_error_of_equations_produced.pdf')

plt.savefig(filename)
