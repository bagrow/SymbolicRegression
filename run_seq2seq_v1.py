"""Train seq2seq model (input=dataset, output=equation) to do
symbolic regression.
"""

from seq2seq_first import seq2seq
from CmaesTrainsNn import CmaesTrainsNn

import numpy as np

s2s = seq2seq(num_encoder_tokens=2,
              primitive_set=['*', '+'],
              terminal_set=['x0'],
              max_decoder_seq_length=10)

x = np.linspace(-1, 1, 20)[None, :]
f = lambda x: x[0]**4 + x[0]**3
y = f(x)

f_test = lambda x: x[0]
y_test = f_test(x)

rep = 1
exp = 0

fitter = CmaesTrainsNn(rep=rep, exp=exp, model=s2s,
                       x=x, y=y,
                       x_test=x, y_test=y_test, test_dataset_name='test')

cmaes_options = {'popsize': 100,
                 'tolfun': 0}  # toleration in function value

fitter.fit(max_FLoPs=10**9, sigma=0.5, cmaes_options=cmaes_options)