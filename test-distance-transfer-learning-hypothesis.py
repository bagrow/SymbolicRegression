#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test-distance-transfer-learning-hypothesis.py
# Jim Bagrow
# Last Modified: 2020-02-07

import scipy.stats

order = ['Koza-1', 'Koza-2', 'Koza-3', 'Nguyen-1', 'Nguyen-3', 'Nguyen-4']

# average final test error
T = {'Nguyen-4': 0.8131568716484233,
     'Koza-1': 0.4510446926864738,
     'Koza-2': 0.3235841595658396,
     'Koza-3': 0.4095771357209498,
     'Nguyen-1': 0.4586843829966759,
     'Nguyen-3': 0.6522588283476058}

# integral values 
d = {'Koza-1': 0.666666666667,
     'Koza-2': 0.666666666667,
     'Koza-3': 1.0,
     'Nguyen-1': 0.548361657292,
     'Nguyen-3': 0.903097970065,
     'Nguyen-4': 0.952380952381}

# put T and d in the same order
T = [T[key] for key in order]
d = [d[key] for key in order]

print(scipy.stats.spearmanr(T,d))

"""
Results:
SpearmanrResult(correlation=0.1449427589131121, pvalue=0.7841083696021083)
"""
