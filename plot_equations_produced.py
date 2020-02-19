from GeneticProgramming.Tree import Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def check_cycle(seq, cycle_start, cycle_length):
    """Check if there is a cyclic pattern in seq
    starting at index cycle_start of length cycle_length.

    Checks that seq[i] == seq[i+cycle_length] for all i.

    Parameters
    ----------
    seq : list
        The list of things that we think might contain
        a cycle
    cycle_start : int
        The index where we think the cycle starts.
    cycle_length : int
        The length of the cycle.

    Returns
    -------
     : bool
        True if cycle.
    """

    for i in range(cycle_start,len(seq)):

        if i+cycle_length < len(seq):
            if seq[i] != seq[i+cycle_length]:

                return False

    return True


def find_cycle(seq):
    """Find any cycles in seq.

    Parameters
    ----------
    seq : list
        The list of things that we think might contain
        a cycle

    Returns
    -------
    cycle_start : int
        The index where the cycle starts or None
        if no cycle found.
    cycle_length : int
        The length of the cycle or None if no cycle
        found.

    Examples
    --------
    >>> seq = [3, 2, 1, 2, 1, 2]
    >>> cycle_start, cycle_length = find_cycle(seq)
    1, 2

    >>> seq = [1, 1, 1, 1, 1, 1]
    >>> cycle_start, cycle_length = find_cycle(seq)
    0, 1
    
    >>> seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    >>> cycle_start, cycle_length = find_cycle(seq)
    0, 3
    """

    for i, s in enumerate(seq):

        try:
            j = seq[i+1:].index(s)
            cycle_start = i
            cycle_length = j+1

            if check_cycle(seq=seq,
                           cycle_start=cycle_start, 
                           cycle_length=cycle_length):
                return cycle_start, cycle_length

        except ValueError:
            # No s found in seq[s+1:]
            pass

    return None, None


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
max_t = 22
len_t = []

for rep in range(19,30):

    for test_index in range(6):

        plt.close('all')
        fig, axes = plt.subplots(ncols=max_t, nrows=7,
                                 sharey=True, sharex=True, figsize=(15,4))

        for f_name_index, f_name in enumerate(function_order):

            f = eval('lambda x: '+function_strs[f_name])

            all_rewrite_data = df.loc[(df['rep'] == rep) & (df['dataset'] == f_name) & (df['test_index'] == test_index)]
            len_t.append(len(all_rewrite_data))

            cycle_start, cycle_length = find_cycle(list(all_rewrite_data['f_hat_t'].values))

            for t in range(max_t):

                plt.sca(axes[f_name_index, t])

                if cycle_start is not None:
                    if t >= cycle_start+cycle_length:
                        plt.axis('off')
                        continue

                    if t == cycle_start:
                        ax = plt.gca()
                        ax.spines['bottom'].set_color('#999999')
                        ax.spines['top'].set_color('#999999') 
                        ax.spines['right'].set_color('#999999')
                        ax.spines['left'].set_color('#999999')

                row = all_rewrite_data.loc[df['t'] == t]

                # end up here is no cycles and not
                # longest rewrite seqeuence
                if len(row) == 0:
                    plt.axis('off')
                    continue

                x = np.linspace(-1, 1, 20)

                if len(row['f_hat_t']) > 1:
                    print('row is more than one row! It is', len(row['f_hat_t']), 'rows.')
                    exit()

                if row['f_hat_t'].iloc[:].values[0] == 'None':
                    continue

                tree = Tree(tree=row['f_hat_t'].iloc[:].values[0])
                f_hat_t = eval('lambda x: '+tree.convert_lisp_to_standard_for_function_creation())

                plt.plot(x, f_hat_t(x[None, :]), color=color_dict[f_name], label='$\\hat{f}_t$', lw=0.75)
                plt.plot(x, f(x[None, :]), '--', color=color_dict[f_name], label='$f$', lw=0.75)

                if f_name_index < 1:
                    plt.title('$t = '+str(t)+'$')

                if f_name_index == 6 and t == 0:
                    plt.xlabel('$x$')
                    plt.ylabel('$y$')

        labels = list(color_dict.keys())
        colors = color_dict.values()
        custom_lines = [plt.Line2D([0], [0], color=c) for c in colors]
        leg1 = fig.legend(custom_lines, labels,
                          ncol=9, handlelength=1,
                          handletextpad=0.4, labelspacing=0.25,
                          columnspacing=1.3,
                          loc='center', bbox_to_anchor=(0.6, 0.94),
                          fontsize=8, title='Line Colors',
                          title_fontsize=9)

        labels = ['$\\hat{f}_t$', '$f$']
        custom_lines = [plt.Line2D([0], [0], color='k'), plt.Line2D([0], [0], color='k', dashes=(2,1))]

        leg2 = fig.legend(custom_lines, labels,
                          ncol=9, handlelength=1,
                          handletextpad=0.4, labelspacing=0.25,
                          columnspacing=1.3,
                          loc='center', bbox_to_anchor=(0.2, 0.94),
                          fontsize=8, title='Line Types',
                          title_fontsize=9)

        plt.gca().add_artist(leg1)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.12, wspace=0.15, hspace=0.2)

        filename = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(exp), 'figures', 'equations', 'plot_equations_produced_rep'+str(rep)+'_test_index'+str(test_index)+'.pdf')

        plt.savefig(filename)

        print('.', flush=True, end='')
print(len_t)
print(max(len_t))