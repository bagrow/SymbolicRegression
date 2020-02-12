#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test-distance-transfer-learning-hypothesis.py
# Jim Bagrow
# Last Modified: 2020-02-07

import scipy.stats

#Order = ["Koza-1", Koza-2, Koza-3, Nguyen-1, Nguyen-3, Nguyen-4]

# average final test error 
T = [0.6030790859826828, 0.44561392892601276, 0.5346375142720746,
     0.4417043727167457, 0.8499866633253066, 1.0020545267637393]

# integral values 
d= [0.666666666667, 0.666666666667, 1.0, 0.548361657292, 0.903097970065,
    0.952380952381]

print(scipy.stats.spearmanr(T,d))

"""
Results:
SpearmanrResult(correlation=0.6087595874350709, pvalue=0.19966018946036257)
"""
