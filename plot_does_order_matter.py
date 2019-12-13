from symbolic_regression_game import SymbolicRegressionGame
from Tree import Tree
from protected_functions import *

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt

import os
import argparse


def play_game(env, model):

    obs = env.reset()
    state = obs
    done = False

    for _ in range(time_limit):
        # We need to pass the previous state and a mask for recurrent policies
        # to reset lstm state when a new episode begin
        action, state = model.predict(obs, state=state, mask=done)
        obs, reward , done, _ = env.step(action)

    # return the error
    return env.env_method('get_error', 'train')[0]


exp = 23

# keijzer11
benchmark_index = 3

target_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                'keijzer15', 'r1', 'r2', 'r3']

lisps = {'quartic': '(* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))',
         'septic': '(* (x0) (+ (1) (* (x0) (+ (-2) (* (x0) (+ (1) (* (x0) (+ (-1) (* (x0) (+ (1) (* (x0) (+ (-2) (x0)))))))))))))',
         'nonic': '(* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))))))))))))',
         'r1': '(% (* (+ (x0) (1)) (* (+ (x0) (1)) (+ (x0) (1)))) (+ (1) (* (x0) (+ (-1) (x0)))))',
         'r2': '(% (+ (* (* (* (x0) (x0)) (x0)) (- (* (x0) (x0)) (3))) (1)) (+ (* (x0) (x0)) (1)))',
         'r3': '(% (* (* (* (* (x0) (x0)) (x0)) (* (x0) (x0))) (+ (1) (x0))) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))))',
         'keijzer11': '(+ (* (x0) (x1)) (sin (* (- (x0) (1)) (- (x1) (1)))))',
         'keijzer12': '(+ (* (* (* (x0) (x0)) (x0)) (- (x0) (1))) (* (x1) (- (* (0.5) (x1)) (1))))',
         'keijzer13': '(* (6) (* (sin (x0)) (cos (x1))))',
         'keijzer14': '(% (8) (+ (2) (+ (* (x0) (x0)) (* (x1) (x1)))))',
         'keijzer15': '(+ (* (x0) (- (* (0.2) (* (x0) (x0))) (1))) (* (x1) (- (* (0.5) (* (x1) (x1))) (1))))'}

jumbled_target_name_indices = [(benchmark_index+i+1) % len(target_names)  for i, _ in enumerate(target_names)] 
jumbled_target_name = [target_names[i] for i in jumbled_target_name_indices]

sorted_errors = []
shuffled_errors = []

for rep in range(11):

    seed = 100*exp+rep
    rng = np.random.RandomState(seed)
    v = rng.uniform(-5, 5)

    tree = Tree(rng=None, tree='(+ '+lisps[jumbled_target_name[-1]]+' (0))', num_vars=2)
    f = eval('lambda x, v: '+tree.convert_lisp_to_standard_for_function_creation()+'+v')
    x = np.vstack((np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)))
    y = f(x, v)

    save_loc = os.path.join(os.environ['EE_DATA'], 'experiment'+str(exp))

    time_limit = 10

    env = DummyVecEnv([lambda: SymbolicRegressionGame(rng=rng, x=x, y=y, f=f, time_limit=time_limit, tree=tree,
                                                      target=jumbled_target_name[-1])])

    model_save_loc = os.path.join(save_loc, 'trained_models', 'vshift_mlp_model_rep'+str(rep))

    # Load saved model
    model = PPO2.load(model_save_loc, env=env)

    num_games = 100

    sorted_errors.extend([play_game(env, model) for _ in range(num_games)])

    # shuffle x-values
    rng.shuffle(x[0])

    shuffled_errors.extend([play_game(env, model) for _ in range(num_games)])

plt.figure()
plt.boxplot([sorted_errors, shuffled_errors], labels=['Training Order', 'Shuffled Order'])
plt.ylabel('Final Root Mean Squared Error')
plt.title('Does order of $x$-values matter?')

stat, pvalue = scipy.stats.mannwhitneyu(sorted_errors, shuffled_errors, alternative='less')
print('p-value', pvalue)

plt.savefig(os.path.join(save_loc, 'figures', 'plot_does_order_matter.pdf'))