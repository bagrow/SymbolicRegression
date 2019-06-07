from GeneticProgrammingAfpo.protected_functions import *

import numpy as np

def koza1(x): return x**4 + x**3 + x**2 + x
def koza2(x): return x**5 - 2 * x**3 + x
def koza3(x): return x**6 - 2 * x**4 + x**2

def nguyen1(x): return x**3 + x**2 + x
def nguyen3(x): return x**5 + x**4 + x**3 + x**2 + x
def nguyen4(x): return x**6 + x**5 + x**4 + x**3 + x**2 + x
def nguyen5(x): return np.sin(x**2) * np.cos(x) - 1
def nguyen6(x): return np.sin(x) + np.sin(x + x**2)
def nguyen7(x): return plog(x + 1) + plog(x**2 + 1)
def nguyen8(x): return psqrt(x)

def keijzer1(x): return 0.3 * x * np.sin(2 * np.pi * x)
def keijzer2(x): return 0.3 * x * np.sin(2 * np.pi * x)
def keijzer4(x): return x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1)

def scaled_sinc(x): return 5 * pdivide(np.sin(x), x)
def automatic_french_curve(x): return 4.26 * (np.exp(-x) - 4 * np.exp(-2 * x) + 3 * np.exp(-3 * x))
def chebyshev_polynomial(x): return 3 * np.cos(3 * np.arccos(x))
def ripple(x): return (x[0] - 3.) * (x[1] - 3.) + 2. * np.sin(x[0] - 4.) * (x[1] - 4.)
def rat_pol_3d(x): return 30. * (x[0] - 1.) * (x[2] - 1.) / x[1]**2 / (x[0] - 10.)
def u_ball_5d(x): return 10. / (5. + (x[0]-3.)**2 + (x[1]-3.)**2 + (x[2]-3.)**2 + (x[3]-3.)**2 + (x[4]-3.)**2)
def identity(x): return x
def paige1(x): return 1/(1+x[0]**(-4)) + 1/(1+x[1]**(-4))
def fr_test1(x): return np.sin(pdivide(x[0], x[1], value=0.))

# convert constants to intervals
def make_interval(const): return 'interval('+const+')'
