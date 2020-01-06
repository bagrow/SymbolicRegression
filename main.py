import GeneticProgramming as GP
from GeneticProgramming.protected_functions import *
# from get_computation import get_computation_time, get_time
from symbolic_regression_game import SymbolicRegressionGame

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers import Input, SimpleRNN

import argparse
import os
import time
import itertools


def train_equation_engineer(rep, exp, timeout, fixed_adjustments, shift, scale, horizontal, vertical,
                            benchamrk_datasets, dataset_name, max_adjustment):
    """Train equation engineer (EE)

    Parameters
    ----------
    rep : int
        Repetition number. Partially determines the seed. Also effects output filenames.
    exp : int
        Experiment number. Partitally determines the seed. Also effects saved file locations.
    timeout : int
        Max ammount of time allowed for training.
    fixed_adjustemnts : bool
        If true the constants specified to shift/scale are not changed.
    shift : bool
        If true, the target function is a shift of base function.
    scale : bool
        If true, the target function is a scale of the base function.
    horizontal : bool
        If true, the scale/shift is done horizontally.
    vertical : bool
        If true, the scale/shift is done vertically.
    benchmark_datasets : dict
        The datasets are the value and the keys are the names of the benchmark functions.
    dataset_name : str
        The name of the benchmark function that will be used for testing
    max_adjustment : dict
        The maximum allowed shift/scale.
    """

    sigma = 0.5
    function_evals = float('inf')
    seed = 100*args.exp + args.rep + 1

    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    num_input = 3
    num_output = 2 if args.extra_constant else 1
    num_data_points = 2

    model = Sequential()
    model.add(SimpleRNN(num_output, input_shape=(num_data_points, num_input),
                        return_sequences=False, activation='tanh'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    # get number of weights
    num_weights = 0
    for layer in model.layers:

        layer_weights = layer.get_weights()

        # Things like input layer have length 0.
        if len(layer_weights) > 0:
            num_weights += np.prod(layer_weights[0].shape)

            if len(layer_weights) == 3:
                num_weights += np.prod(layer_weights[1].shape)

    global best

    # best = (error, weights, model)
    best = (float('inf'), None, None)

    num_targets = 1
    num_base_function_per_target = 1
    num_test_targets = 1
    depth = 6

    # get data
    num_ops_train, datasets = get_data_for_equation_corrector(rng=rng,
                                                              num_targets=num_targets,
                                                              num_base_function_per_target=num_base_function_per_target,
                                                              depth=depth,
                                                              max_adjustment=max_adjustment,
                                                              shift=shift,
                                                              scale=scale,
                                                              horizontal=horizontal,
                                                              vertical=vertical,
                                                              fixed_adjustments=fixed_adjustments)

    num_ops_val, datasets_validation_functions = get_data_for_equation_corrector(rng=rng,
                                                                                 num_targets=num_targets,
                                                                                 num_base_function_per_target=num_base_function_per_target,
                                                                                 depth=depth,
                                                                                 max_adjustment=max_adjustment,
                                                                                 shift=shift,
                                                                                 scale=scale,
                                                                                 horizontal=horizontal,
                                                                                 vertical=vertical,
                                                                                 fixed_adjustments=fixed_adjustments)


    all_datasets = {'training': datasets,
                    'validation': datasets_validation_functions,
                    'testing': benchamrk_datasets[dataset_name]}

    save_loc = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(exp))

    os.makedirs(save_loc, exist_ok=True)

    # Make a file for parameter summary
    summary_data = [('experiment', exp),
                    ('number of inputs', num_input),
                    ('number of outputs', num_output),
                    ('number of target functions', num_targets),
                    ('number of target functions for testing', num_test_targets),
                    ('max depth', depth),
                    ('sigma (CMA-ES)', sigma),
                    ('max number of function evaluations', function_evals),
                    ('horizontal', horizontal),
                    ('max shift', max_adjustment),
                    ('timeout', timeout),
                    ('fixed adjustment', fixed_adjustments)]

    df = pd.DataFrame(summary_data, columns=['Parameters', 'Values'])
    df.to_csv(os.path.join(save_loc, 'summary_exp'+str(exp)+'_'+dataset_name+'.csv'))

    # get number of operations to evaluate training and validation datasets
    num_ops_per_tree = num_ops_train + num_ops_val

    cmaes_options = {'maxfevals': function_evals,
                     'ftarget': 1e-30,
                     # 'tolfun': 0,   # toleration in function value
                     # 'tolfunhist': 0, # tolerance in function value history
                     'popsize': 100,
                     'seed': seed,
                     'verb_log': 0,
                     'timeout': timeout}

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

    jumbled_target_name_indices = [(args.benchmark_index+i+1) % len(target_names)  for i, _ in enumerate(target_names)] 
    jumbled_target_name = [target_names[i] for i in jumbled_target_name_indices]
    print(target_names)
    print(jumbled_target_name)

    if args.extra_constant:

        tree_string = '(+ (* (1) '+lisps[jumbled_target_name[0]]+') (0))'
        tree = GP.Tree(rng=None, tree=tree_string, num_vars=2)
        f_string = 'lambda x, v, s: s*('+tree.convert_lisp_to_standard_for_function_creation()+')+v'

        tree_test_string = '(+ (* (1) '+lisps[jumbled_target_name[-1]]+') (0))'
        tree_test = GP.Tree(rng=None, tree=tree_test_string, num_vars=2)
        f_test_string = 'lambda x, v, s: s*('+tree_test.convert_lisp_to_standard_for_function_creation()+')+v'

    else:

        tree_string = '(+ '+lisps[jumbled_target_name[0]]+' (0))'
        tree = GP.Tree(rng=None, tree=tree_string, num_vars=2)
        f_string = 'lambda x, v: '+tree.convert_lisp_to_standard_for_function_creation()+'+v'

        tree_test_string = '(+ '+lisps[jumbled_target_name[-1]]+' (0))'
        tree_test = GP.Tree(rng=None, tree=tree_test_string, num_vars=2)
        f_test_string = 'lambda x, v: '+tree_test.convert_lisp_to_standard_for_function_creation()+'+v'

    f_test = eval(f_test_string)
    x_test = np.vstack((np.linspace(-1, 1, num_data_points), np.linspace(-1, 1, num_data_points)))

    if args.extra_constant:
        y_test = f_test(x_test, 5, 0.5)

    else:
        y_test = f_test(x_test, 5)

    # get the tree, +0 for the v
    f = eval(f_string)
    x = np.vstack((np.linspace(-1, 1, num_data_points), np.linspace(-1, 1, num_data_points)))

    rng = np.random.RandomState(seed)
    v = rng.uniform(-5, 5)
    
    if args.extra_constant:
        s = rng.uniform(-5, 5)
        y = f(x, v, s)

    else:
        y = f(x, v)

    time_limit = 10

    EE = SymbolicRegressionGame(rng=rng, x=x, f=f, y=y, time_limit=time_limit, tree=tree,
                                f_test=f_test, x_test=x_test, y_test=y_test,
                                target=jumbled_target_name[-1], model=model,
                                extra_constant=args.extra_constant)

    best_individual_data = [['Generation', 'Sum Train Error Sum','Test Error Sum', 'Number of Floating Point Operations']]
    gen = 0
    FLoPs_checkpoint = 0
    target_index = 1

    weights = rng.uniform(-1, 1, size=num_weights)
    es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

    mu = es.sp.weights.mu
    popsize = es.popsize

    # get CMA-ES ops
    n = len(weights)

    # eq: 5, 9, y, 24, 30, c_1, c_mu, 31, 37
    cma_ops_per_gen = (2*n) + 2*(mu-1)*n + (n+1) + (2*n+7) + (2+4*mu+4*n**2+n) + (2) + (3) + (6 + 5*n**2) + (2*n+5)

    while max_FLoPs >= EE.FLoPs:

        pop_data_summary = []

        while not es.stop():

            v = rng.uniform(-5, 5)
            s = rng.uniform(-5, 5)

            solutions = es.ask()
            evaluate_pop = [EE.play_game(weights=w, v=v, s=s) for w in solutions]

            es.tell(solutions, evaluate_pop)
            es.disp()

            best_index = np.argmin(evaluate_pop)

            if evaluate_pop[best_index] < best[0]:
                EE.set_weights(solutions[best_index])
                best = (evaluate_pop[best_index], solutions[best_index], EE.model)
                print('new best', best[0])

            gen += 1

            # update number of operations
            EE.FLoPs += cma_ops_per_gen

            # save best individuals during training
            test_error = EE.play_game(weights=best[1], v=v, s=s, datatype='test', final_error_only=True)
            best_individual_data.append([gen,
                                         best[0],
                                         test_error,
                                         EE.FLoPs])

            print('total compute', EE.FLoPs)

            # check if max number of computations have occured
            if max_FLoPs < EE.FLoPs:
                break

            if EE.FLoPs - FLoPs_checkpoint > max_FLoPs/(len(target_names)-1):

                # update training data but not testing data
                tree = GP.Tree(rng=None, tree='(+ '+lisps[jumbled_target_name[target_index]]+' (0))', num_vars=2)

                if EE.extra_constant:
                    f = eval('lambda x, v, s: s*('+tree.convert_lisp_to_standard_for_function_creation()+')+v')

                else:
                    f = eval('lambda x, v: '+tree.convert_lisp_to_standard_for_function_creation()+'+v')

                EE.tree = tree
                EE.f = f
                
                FLoPs_checkpoint += max_FLoPs/(len(target_names)-1)
                print('checkpoint', jumbled_target_name[target_index])
                target_index += 1

        es.result_pretty()

        # restart cma-es with different seed
        weights = rng.uniform(-1, 1, size=num_weights)
        cmaes_options['seed'] += 10000
        es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

    df = pd.DataFrame(best_individual_data[1:], columns=best_individual_data[0])
    df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(rep)+'_'+dataset_name+'.csv'), index=False)

    df = pd.DataFrame(pop_data_summary, columns=['Generation', 'Mean Training Error', 'Mean Validation Error'])
    df.to_csv(os.path.join(save_loc, 'pop_data_summary_rep'+str(rep)+'_'+dataset_name+'.csv'), index=False)

    best[2].save(os.path.join(save_loc, 'best_ind_model_rep'+str(rep)+'_'+dataset_name+'.csv'))


def get_benchmark_datasets(rng, max_adjustment, shift, scale, horizontal, vertical, jumbled_target_name):

    target_strings = {'quartic': 'x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0])))',
                      'septic': 'x[0] * (1 - x[0] * (2 - x[0] * (1 - x[0] * (1 - x[0] * (1 - x[0] * (2 - x[0]))))))',
                      'nonic': 'x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0] * (1 + x[0]))))))))',
                      'keijzer11': '(x[0] * x[1]) + np.sin((x[0] - 1) * (x[1] - 1))',
                      'keijzer12': 'x[0] ** 4 - x[0] ** 3 + (x[1] ** 2 / 2.0) - x[1]',
                      'keijzer13': '6 * np.sin(x[0]) * np.cos(x[1])',
                      'keijzer14': '8.0 / (2 + x[0] ** 2 + x[1] ** 2)',
                      'keijzer15': '(x[0] ** 3 / 5.0) + (x[1] ** 3 / 2.0) - x[0] - x[1]',
                      'r1': '((x[0] + 1) ** 3) / (x[0] ** 2 - x[0] + 1)',
                      'r2': '(x[0] ** 5 - (3 * (x[0] ** 3)) + 1) / (x[0] ** 2 + 1)',
                      'r3': '(x[0] ** 6 + x[0] ** 5) / (x[0] ** 4 + x[0] ** 3 + x[0] ** 2 + x[0] + 1)'}

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

    benchamrk_datasets = {}

    for f_name, f_func in target_strings.items():

        x0_train = rng.uniform(-1, 1, 20)
        x1_train = rng.uniform(-1, 1, 20)

        input_train = np.vstack((x0_train, x1_train))

        x0_val = rng.uniform(-1, 1, 20)
        x1_val = rng.uniform(-1, 1, 20)

        input_val = np.vstack((x0_val, x1_val))

        x0_test = rng.uniform(-1, 1, 1000)
        x1_test = rng.uniform(-1, 1, 1000)

        input_test = np.vstack((x0_test, x1_test))

        testing = f_name == jumbled_target_name[-1]

        f, _ = get_function(rng=rng, f_str=f_func, max_adjustment=max_adjustment,
                         shift=shift, scale=scale, horizontal=horizontal, vertical=vertical, testing=testing)

        output_train = f(input_train)
        output_val = f(input_val)
        output_test = f(input_test)

        num_vars = 2 if 'x1' in lisps[f_name] else 1
        dataset = ((GP.Tree(tree=lisps[f_name], num_vars=num_vars),
                   np.vstack([output_train, input_train]).T,
                   np.vstack([output_val, input_val]).T,
                   np.vstack([output_test, input_test]).T),)

        benchamrk_datasets[f_name] = dataset

    return benchamrk_datasets


def get_function(rng, f_str, max_adjustment, shift, scale, horizontal, vertical, testing):

    offset_amounts = {}

    if shift:

        if vertical:

            shift_amount = 0

            shift_amount = rng.uniform(0, max_adjustment['vshift']) if testing is False else 5
            offset_amounts['vshift'] = shift_amount

            f_str = f_str + '+' +str(shift_amount)

        if horizontal:

            shift_amount = rng.uniform(0, max_adjustment['hshift'])

            offset_amounts['hshift'] = shift_amount

            f_str = f_str.replace('x[0]', 'np.add(x[0],'+str(shift_amount)+')')

    if scale:

        if vertical:

            scale_amount = rng.uniform(0, max_adjustment['vscale'])
            offset_amounts['vscale'] = scale_amount

            f_str = 'np.multiply('+f_str+','+str(scale_amount)+')'

        if horizontal:

            scale_amount = rng.uniform(0, max_adjustment['hscale'])
            offset_amounts['hscale'] = scale_amount

            f_str = f_str.replace('x[0]', 'np.multiply(x[0],'+str(scale_amount)+')')

    lambda_string = 'lambda x: '+f_str

    f = eval(lambda_string)

    return f, offset_amounts


def get_data_for_equation_corrector(rng, num_targets, num_base_function_per_target,
                                    depth, max_adjustment=50, shift=False, scale=False,
                                    horizontal=False, vertical=False,
                                    fixed_adjustments=False):
    """Generate a dataset for multiple functions. Keep track of
    the base function(s) connected with the dataset.

    Parameters
    ----------
    rng : random number generator
        Example: np.random.RandomState(0)
    num_targets : int
        The number of target functions.
    num_base_function_per_target : int
        The number of base functions to make for
        each target funtion.
    depth : int
        The max depth of the trees (that represent
        the target functions).

    Returns
    -------
    datasets : list
        List of tuple containnig (base_function, dataset).
    """

    primitives = ['*', '+', '-', 'sin', ]
    terminals = ['#x', '#f']

    # Pick the number of input variables
    # between 1 and 6.
    num_vars = 1    #rng.choice(5)+1

    number_of_operations = 0

    targets = []
    trees = ['(- (* (* (x0) (x0)) (x0)) (x0))',
             '(* (x0) (x0))',
             '(sin (x0))',
             '(* (sin (x0)) (cos (x0)))',
             '(+ (x0) (3))',
             '(* (x0) (sin (x0)))']
    datasets = []
    j = 0

    while len(datasets) < num_targets:

        tree = trees[j]
        j += 1

        t = GP.Individual(rng=rng, primitive_set=primitives, terminal_set=terminals, num_vars=num_vars,
                          tree=tree)#, depth=depth, method='grow')

        f = eval('lambda x: ' + t.convert_lisp_to_standard_for_function_creation())
        outputs = f(np.array([np.linspace(-1, 1, 1000)]))

        while 'x0' not in t.get_lisp_string() or len(np.unique(np.around(outputs,7))) == 1 or np.any([t.get_lisp_string() == ind.get_lisp_string() for ind in targets]) or t.get_lisp_string() == '(x0)':

            t = GP.Individual(rng=rng, primitive_set=primitives, terminal_set=terminals, num_vars=num_vars,
                              depth=depth, method='grow')

            f = eval('lambda x: ' + t.convert_lisp_to_standard_for_function_creation())
            outputs = f(np.array([np.linspace(-1, 1, 1000)]))

        targets.append(t)

        num_leaves, num_nodes = t.get_num_leaves(return_num_nodes=True)
        number_of_operations += num_nodes-num_leaves
        
        if vertical:
            number_of_operations += 1

        if horizontal:
            number_of_operations += t.get_lisp_string(actual_lisp=True).count('x0')

        if fixed_adjustments:
            rand_offset = rng.randint

        else:
            rand_offset = rng.uniform

        base_function_string = t.convert_lisp_to_standard_for_function_creation()

        function, offset_amounts = get_function(rng=rng, f_str=base_function_string, max_adjustment=max_adjustment,
                                                shift=shift, scale=scale, horizontal=horizontal, vertical=vertical,
                                                testing=False)  # doesn't matter here just getting tree sizes

        # Make inputs
        x = np.array([rng.uniform(-1, 1, size=60) for _ in range(num_vars)]).T

        training_indices = rng.choice(60, size=20, replace=False)
        remaining_indices = [i for i in range(60) if i not in training_indices]
        validation_indices = rng.choice(remaining_indices, size=20, replace=False)
        testing_indices = np.array([i for i in range(60) if i not in training_indices and i not in validation_indices])

        x_training = x[training_indices]
        x_validation = x[validation_indices]
        x_testing = x[testing_indices]

        training_dataset = np.hstack((np.array([function(x_training.T)]).T, x_training))
        validation_dataset = np.hstack((np.array([function(x_validation.T)]).T, x_validation))
        testing_dataset = np.hstack((np.array([function(x_testing.T)]).T, x_testing))

        target = eval('lambda x: ' + base_function_string)

        if np.all(training_dataset[:, 0] - target(training_dataset[:, 1:].T) == 0):
            print('continuing', len(datasets))
            continue

        datasets.append((GP.Tree(tree), training_dataset, validation_dataset, testing_dataset, offset_amounts))

    return number_of_operations, datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # this will act as an index for rep_list (offset by 1 though)
    parser.add_argument('rep', help='Number of runs already performed', type=int)
    parser.add_argument('exp', help='Experiment number. Used in save location', type=int)

    parser.add_argument('-t', '--timeout', help='The number of seconds use for training based on clock speed.',
                        type=float, action='store', default=float('inf'))

    parser.add_argument('-gp', '--genetic_programming', help='Compare with GP',
                        action='store_true')

    # parser.add_argument('-vs', '--variant_string', help='The string that describes NN options',
    #                     type=str)
    parser.add_argument('-b', '--benchmark_index', help='Sets the benchmark to try to solve.',
                        type=int)
    
    parser.add_argument('--horizontal', help='If True, NN will be trained to do horizontal stuff',
                        action='store_true')
    parser.add_argument('--vertical', help='If True, NN will be trained to do vertical stuff',
                        action='store_true')

    parser.add_argument('--shift', help='Stuff=shifts',
                        action='store_true')
    parser.add_argument('--scale', help='Stuff=scale',
                        action='store_true')

    parser.add_argument('--extra_constant',
                        action='store_true')

    args = parser.parse_args()
    print(args)

    assert (args.shift or args.scale) and (args.horizontal or args.vertical), '--vertical or --horizontal must be used with --shift or --scale'

    max_FLoPs = 10**8

    max_adjustment = {'hshift': 5,
                      'vshift': 5,
                      'hscale': 5,
                      'vscale': 5}

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

    jumbled_target_name_indices = [(args.benchmark_index+i+1) % len(target_names)  for i, _ in enumerate(target_names)] 
    jumbled_target_name = [target_names[i] for i in jumbled_target_name_indices]
    target_name = target_names[args.benchmark_index]

    benchamrk_datasets = get_benchmark_datasets(rng=np.random.RandomState(args.rep+100*args.exp),
                                                max_adjustment=max_adjustment,
                                                shift=args.shift,
                                                scale=args.scale,
                                                horizontal=args.horizontal,
                                                vertical=args.vertical,
                                                jumbled_target_name=jumbled_target_name)

    if args.genetic_programming:

        print('genetic programming')

        assert args.benchmark_index is not None, 'If using genetic programming, must specify --benchmark_index (-b)'

        assert 0 <= args.benchmark_index < len(benchamrk_datasets), '--benchmark_index (-b) too large or too small'

        primitive_set = ['*', '+', '%', '-', 'sin', 'cos']
        terminal_set = ['#x', '#f']

        # Now do this for longer for the test function
        test_function = target_names[args.benchmark_index]

        # get the name from list rather than benchmark_datasets,
        # which is a dict
        rng = np.random.RandomState(args.rep+100*args.exp)

        path = os.path.join(os.environ['GP_DATA'], 'equation_adjuster', 'experiment'+str(args.exp))

        timeout = args.timeout

        # the population from the previous run of
        # genetic programming
        prev_pop = None

        for index in jumbled_target_name_indices:

            function = target_names[index]

            # do this one last since it is the test function
            if function == target_names[args.benchmark_index]:
                continue

            # dataset is the training dataset and validation dataset
            dataset = [benchamrk_datasets[function][0][1], benchamrk_datasets[function][0][2]]
            test_data = benchamrk_datasets[test_function][0][3]

            # get output_path, put target function being trained and
            # put the function that is to be the test.
            output_path = os.path.join(path, 'gp', target_names[args.benchmark_index], function)
            output_file = 'fitness_data_rep' + str(args.rep) + '.csv'

            num_vars = 2 if 'x1' in lisps[function] else 1

            params = {'T': timeout,
                      'given_individual': lisps[function],
                      'max_compute': max_FLoPs/(len(benchamrk_datasets)-1)}

            gp = GP.GeneticProgrammingAfpo(rng=rng,
                                           pop_size=100,
                                           max_gens=60000,
                                           primitive_set=primitive_set,
                                           terminal_set=terminal_set,
                                           # this is not data, which is passed
                                           data=dataset,
                                           test_data=test_data,
                                           prob_mutate=1,
                                           prob_xover=0,
                                           num_vars=num_vars,
                                           mutation_param=2,
                                           # parameters below
                                           **params)

            if prev_pop is not None:
                gp.pop = prev_pop

            # print('before errors', [p.fitness[0] for p in gp.pop])

            info = gp.run(rep=args.rep,
                          output_path=output_path,
                          output_file=output_file)

            # print('after errors', [p.fitness[0] for p in gp.pop])

            prev_pop = gp.pop

    else:
        print('equation adjuster')
        assert args.timeout is not None, 'Specify a time limit with -t or --timeout'

        train_equation_engineer(rep=args.rep, exp=args.exp, timeout=args.timeout,
                                fixed_adjustments=False, shift=args.shift,
                                scale=args.scale, horizontal=args.horizontal,
                                vertical=args.vertical,
                                benchamrk_datasets=benchamrk_datasets,
                                dataset_name=target_name,
                                max_adjustment=max_adjustment)
