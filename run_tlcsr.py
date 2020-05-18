"""Train seq2seq model (input=dataset and previous equation, output=equation) to do
symbolic regression.
"""

from TlcsrNetwork import TlcsrNetwork
from Tlcsr import Tlcsr
# import GeneticProgramming as GP
from GeneticProgramming.IndividualManyTargetData import IndividualManyTargetData
from GeneticProgramming.GeneticProgrammingAfpoManyTargetData import GeneticProgrammingAfpoManyTargetData
from GeneticProgramming.GeneticProgrammingAfpo import GeneticProgrammingAfpo

import create_dataset_from_function as cdff

import numpy as np
import pandas as pd
import pmlb

import argparse
import os
import copy
import itertools
import operator

def get_x_y_data(targets, key):

	if 'f' in targets[key]:

		if targets[key]['spacing'] == 'random':
			x = np.array([[np.random.uniform(targets[key]['a'], targets[key]['b']) for _ in range(targets[key]['num_inputs'])] for _ in range(targets[key]['num_points'])])

		elif targets[key]['spacing'] == 'uniform':

			a = targets[key]['a']
			b = targets[key]['b']

			step = targets[key]['step']

			if len(step) == 1:
				step = step*targets[key]['num_inputs']

			elif len(step) != targets[key]['num_inputs']:
				print('Either use 1 step or the same number of num_inputs. You specified step =', step, 'but num_inputs =', targets[key]['num_inputs'])
				exit()

			n = (b-a)/np.array(step)

			xs = [np.linspace(a, b, int(round(ni, 0))) for ni in n]

			x = np.array(list(itertools.product(*xs)))

		else:
			print('target spacing '+str(targets[key]['spacing'])+' not implemented')
			exit()

		y = cdff.get_y(x, targets[key]['f'])

	elif 'file' in targets[key]:

		path = os.path.join(os.environ['DATASET_PATH'], targets[key]['file'])

		data = pd.read_csv(path)

		x = data.iloc[:,:-1].values

		# keep the : after -1 to get column vec
		y = data.iloc[:,-1:].values

	elif 'pmlb' in targets[key]:
		pmlb_path = os.path.join(os.environ['DATASET_PATH'], 'pmlb')
		os.makedirs(pmlb_path, exist_ok=True)
		x, y = pmlb.fetch_data(key, return_X_y=True, local_cache_dir=pmlb_path)
		y = y[:,None]

	else:
		print('Not implemented!!! in get_x_y_data')
		exit()

	# sort by x[:,0] then x[:,1] and return indices to perform sort
	ind = np.lexsort((x[:,1], x[:,0]))
	x_sorted = x[ind]
	y_sorted = y[ind]

	get_row_set = lambda col1, col2: set([(*a, *b) for a,b in zip(col1,col2)])
	assert get_row_set(x,y) == get_row_set(x_sorted, y_sorted), 'Not the same dataset (not just reordered)!'

	return x_sorted, y_sorted

parser = argparse.ArgumentParser()

parser.add_argument('rep', help='Number of runs already performed', type=int)
parser.add_argument('exp', help='Experiment number. Used in save location', type=int)
parser.add_argument('--use_kexpressions', action='store_true', help='Use k-expressions to interpret input and output to TLC-SR')
parser.add_argument('--genetic_programming', action='store_true', help='Do genetic programming')
parser.add_argument('--single_target', action='store_true', help='Use only one target function to create train, validation, test datasets')
parser.add_argument('--simultaneous_targets', action='store_true', help='Use multiple target functions for train, validation, test datasets')
parser.add_argument('--use_constants', action='store_true', help='Included constants with two nodes: one for one-hot and other for constant value')
parser.add_argument('--inconsistent_x', action='store_true', help='Random (uniformly dist) selection of x-values')
# parser.add_argument('--shuffle_x', action='store_true', help='Shuffle x-values before input into NN')
parser.add_argument('--use_old_benchmarks', action='store_true', help='Use the benchmark functions for datasets')
parser.add_argument('--use_dim2_benchmarks', action='store_true', help='Use the new dim=2 benchmarks')
parser.add_argument('--use_dim5_benchmarks', action='store_true', help='Use the new dim=5 benchmarks')
parser.add_argument('--test_index', type=int, help='Index of the target function to use to create test dataset. Others are used for training if --simultaneous_targets')
parser.add_argument('--max_rewrites', type=int, help='Set the max number of equation rewrites')
parser.add_argument('--constant_targets', action='store_true', help='Use constant regression datasets.')
parser.add_argument('--lines', action='store_true', help='Use lines for target functions')
parser.add_argument('--num_datasets', type=int, help='set the number of datasets')
parser.add_argument('--patience', type=int, help='set the number of generations to weight without validation error decrease before stopping')
parser.add_argument('--easy_for_gp', action='store_true', help='Using same domain for val and test (gp only)')
parser.add_argument('--equation_summary', action='store_true', help='Save equation data')
parser.add_argument('--quick_gens', action='store_true', help='Make TLC-SR gens run faster for testing.')
parser.add_argument('--use_old_primitives', action='store_true', help='use * - + as primitive set')
parser.add_argument('--activation_function', type=str, default='tanh', help='use this activation function')
parser.add_argument('--short_run', action='store_true', help='Run for small effort')

args = parser.parse_args()
print(args)

assert args.test_index is not None, 'Must use --test_index'

assert (args.simultaneous_targets and args.single_target) == False, 'Cannot use --simultaneous_targets and --single_target at the same time'
assert (args.simultaneous_targets or args.single_target), 'Most use --simultaneous_targets or --single_target'


assert (args.use_old_benchmarks and args.constant_targets) == False, 'Cannot use --use_old_benchmarks and --constant_targets as the same time'

assert (args.lines and args.constant_targets) == False, 'Cannot use --lines and --constant_targets at the same time'

assert (args.lines and args.use_old_benchmarks) == False, 'Cannot use --lines and --use_old_benchmarks at the same time'

# essentially not xor
assert (args.lines) == (args.num_datasets is not None), 'Connot use --lines without --num_datasets'

targets = {'Vladislavleva-1': {'f': lambda x: np.exp(-(x[0]-1)**2)/(1.2 + (x[1]-2.5)**2),
							  'a': -0.2,
							  'b': 4.2,
							  'step': [0.1],
							  'num_inputs': 2,
							  'spacing': 'uniform'},
		   'Vladislavleva-3': {'f': lambda x: np.exp(-x[0])*x[0]**3*np.cos(x[0])*np.sin(x[0])*(np.cos(x[0])*(np.sin(x[0]))**2 - 1)*(x[1]-5),
							  'a': -0.5,
							  'b': 10.5,
							  'step': [0.05, 0.5],
							  'num_inputs': 2,
							  'spacing': 'uniform'},
		   'Vladislavleva-4': {'f': lambda x: 10/(5 + (x[0]-3)**2 + (x[1]-3)**2 + (x[2]-3)**2 + (x[3]-3)**2 + (x[4]-3)**2),
							  'a': -0.25,
							  'b': 6.35,
							  'num_points': 5000,
							  'num_inputs': 5,
							  'spacing': 'random'},
	  	   'Vladislavleva-7': {'f': lambda x: (x[0]-3)*(x[1]-3) + 2*np.sin((x[0]-4)*(x[1]-4)),
							  'a': -0.25,
							  'b': 6.35,
							  'num_points': 1000,
							  'num_inputs': 2,
							  'spacing': 'random'},
  	       'Vladislavleva-8': {'f': lambda x: ((x[0]-3)**4 + (x[1]-3)**3 - (x[1]-3))/((x[1]-2)**4 + 10),
							  'a': -0.25,
							  'b': 6.35,
							  'step': [0.2],
							  'num_inputs': 2,
							  'spacing': 'uniform'},
		   'energy efficiency cooling': {'file': 'energy_efficiency_heating.csv'},
		   'energy efficiency heating': {'file': 'energy_efficiency_heating.csv'},
		   'boston housing': {'file': 'boston_housing.csv'},
		   '648_fri_c1_250_50': {'pmlb': True},
		   '654_fri_c0_500_10': {'pmlb': True},
		   '657_fri_c2_250_10': {'pmlb': True},
		   '687_sleuth_ex1605': {'pmlb': True},
		   '210_cloud': {'pmlb': True},
		   '609_fri_c0_1000_5': {'pmlb': True},
		   '612_fri_c1_1000_5': {'pmlb': True},
		   '656_fri_c1_100_5': {'pmlb': True},
		   '599_fri_c2_1000_5': {'pmlb': True},
		   '663_rabe_266': {'pmlb': True},
		   '519_vinnie': {'pmlb': True},
		   '228_elusage': {'pmlb': True}}

# new_target_order = ['Vladislavleva-4',
# 					'energy efficiency cooling',
# 					'energy efficiency heating',
# 					'boston housing',
# 					'648_fri_c1_250_50', 
# 					'654_fri_c0_500_10', 
# 					'657_fri_c2_250_10']

if args.use_dim5_benchmarks:
	new_target_order = ['687_sleuth_ex1605',
						'210_cloud',
						'Vladislavleva-4',
						'609_fri_c0_1000_5',
						'612_fri_c1_1000_5',
						'656_fri_c1_100_5',
						'599_fri_c2_1000_5']

elif args.use_dim2_benchmarks:
	new_target_order = ['663_rabe_266',
						'Vladislavleva-7',
						'519_vinnie',
						'Vladislavleva-8',
						'Vladislavleva-1',
						'228_elusage',
						'Vladislavleva-3']

# get a single flag to use later
if args.use_dim2_benchmarks or args.use_dim5_benchmarks:
	use_new_benchmarks = True
else:
	use_new_benchmarks = False

if args.use_kexpressions:
	options = {'use_k-expressions': True,
			   'head_length': 15}   # this defines length eq output from NN

else:
	options = None

options['equation_summary'] = args.equation_summary
options['quick_gens'] = args.quick_gens
options['activation_function'] = args.activation_function

timelimit = args.max_rewrites if args.max_rewrites is not None else 100

if args.use_old_benchmarks:
	function_strs = ['x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Nguyen-4
					 'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
					 'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
					 'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
					 'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
					 'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]' # Nguyen-3
					]

	assert 0 <= args.test_index < len(function_strs), '--test_index must be between 0 and '+str(len(function_strs)-1)


	train_function_strs = [f for i, f in enumerate(function_strs) if i != args.test_index]

elif args.constant_targets:

	train_function_strs = [str(c) + '+x[0]*0' for c in np.random.uniform(-10, 10, 5)]

	function_strs = copy.copy(train_function_strs)

	# test: f(x) = 1
	function_strs.insert(0, '1 + x[0]*0')

elif args.lines:

	train_function_strs = [str(c[0]) + '*x[0]+' + str(c[1]) for c in np.random.uniform(-10, 10, size=(args.num_datasets, 2))]

	function_strs = copy.copy(train_function_strs)

	# test: f(x) = 2x + 3
	function_strs.insert(0, '2*x[0] + 3')

else:
	train_function_strs = ['x[0]',
						   '2*x[0]',
						   'x[0]**2',
						   'x[0]**2 + x[0]',
						   'x[0]**3']

if not use_new_benchmarks:
	if args.use_old_benchmarks or args.constant_targets or args.lines:
		f_test_str = function_strs[args.test_index]
		f_test = eval('lambda x: '+f_test_str)

	else:

		test_function_strs = ['x[0]**6 + x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]' # Nguyen-4
							  'x[0]**4 + x[0]**3 + x[0]**2 + x[0]',   # Koza-1
							  'x[0]**5 - 2*x[0]**3 + x[0]',   # Koza-2
							  'x[0]**6 - 2*x[0]**4 + x[0]**2',    # Koza-3
							  'x[0]**3 + x[0]**2 + x[0]', # Nguyen-1
							  'x[0]**5 + x[0]**4 + x[0]**3 + x[0]**2 + x[0]', # Nguyen-3
							 ]

		assert 0 <= args.test_index < len(test_function_strs), '--test_index must be between 0 and '+str(len(test_function_strs)-1)

		f_test_str = test_function_strs[args.test_index]

		f_test = eval('lambda x: '+f_test_str)

if args.single_target:

	if use_new_benchmarks:
		assert 0 <= args.test_index < len(new_target_order), '--test_index must be between 0 and 6 (inclusive) when using --use_new_benchmarks'

		dataset_name = new_target_order[args.test_index]

		x_all, y_all = get_x_y_data(targets, dataset_name)

		# split
		train_size = int(0.7*len(x_all))
		val_size = int((len(x_all)-train_size)/2)
		test_size = len(x_all) - train_size - val_size
		train_indices = np.random.choice(len(x_all), train_size, replace=False)
		not_train_indices = np.array([e for e in range(len(x_all)) if e not in train_indices])
		# print(np.random.choice(len(not_train_indices), val_size, replace=False))
		# print(type(np.random.choice(len(not_train_indices), val_size, replace=False)))
		val_indices = not_train_indices[np.random.choice(len(not_train_indices), val_size, replace=False)]
		test_indices = [e for e in not_train_indices if e not in val_indices]
		# print(train_indices)

		x_train = x_all[train_indices, :]
		y_train = y_all[train_indices, :]

		x_val = x_all[val_indices, :]
		y_val = y_all[val_indices, :]

		x_test = x_all[test_indices, :]
		y_test = y_all[test_indices, :]

		assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1]

		X_train = [x_train]
		Y_train = [y_train]

		# print('train_size', train_size, len(train_indices), x_train.shape, y_train.shape)
		# print('val_size', val_size, len(val_indices), x_val.shape, y_val.shape)
		# print('test_size', test_size, len(test_indices), x_test.shape, y_test.shape)

		# print(x_all.shape)
		# print(y_all.shape)
		# print(len(x_all))
		# exit()

	else:
		X_train = [np.array(sorted(np.random.uniform(-1, 1, size=20)))[:, None]]
		Y_train = [cdff.get_y(x, f_test) for x in X_train]

		x_val = np.array(sorted(np.random.uniform(-1, 1, size=20)))[:, None]
		y_val = cdff.get_y(x_val, f_test)

		x_test = np.array(sorted(np.random.uniform(-1, 1, size=20)))[:, None]

		y_test = cdff.get_y(x_test, f_test)

else:
	if args.inconsistent_x:
		gen_x_values = lambda: np.array(sorted(np.random.uniform(-1, 1, size=20))) 

	else:
		gen_x_values = lambda: np.linspace(-1, 1, 20)

	if not use_new_benchmarks:
		train_functions = [eval('lambda x: '+f_str) for f_str in train_function_strs]

		if not (args.constant_targets or args.lines):
			assert len(train_functions) == len(function_strs) - 1, 'len(functions) != len(function_strs) - 1'

		X_train = [gen_x_values()[:, None] for _ in range(len(train_functions))]
		Y_train = [cdff.get_y(x, f) for x, f in zip(X_train, train_functions)]

		x_val = gen_x_values()[:, None]

		if args.constant_targets:
			f_val_str = str(np.random.uniform(-10,10)) + '+x[0]*0'
		elif args.lines:
			f_val_str = str(np.random.uniform(-10,10)) + '*x[0]+' + str(np.random.uniform(-10,10))
		else:
			f_val_str = 'x[0]**4 + x[0]'
		f_val = eval('lambda x: ' + f_val_str)
		y_val = cdff.get_y(x_val, f_val)

		assert f_val_str != f_test_str, 'val == testing (single_target=False)'
		assert f_val_str not in train_function_strs, 'val is part of training (single_target=False)'

		x_test = gen_x_values()[:, None]

		y_test = cdff.get_y(x_test, f_test)

	else:

		assert 0 <= args.test_index < len(new_target_order)-1, '--test_index must be between 0 and 5 (inclusive) when using --use_dim#_benchmarks'

		# get arbitrary but consistent validation domain
		rng_val = np.random.RandomState(0)
		val_index = rng_val.choice(len(new_target_order))
		val_name = new_target_order[val_index]

		x_val, y_val = get_x_y_data(targets, val_name)
		print('val_name', val_name)
		print(x_val.shape, y_val.shape)

		new_target_order_no_val = [target for target in new_target_order if target != val_name]

		assert len(new_target_order_no_val) == 6, 'Unexpected number of datasets'

		test_name = new_target_order_no_val[args.test_index]

		x_test, y_test = get_x_y_data(targets, test_name)
		print('test_name', test_name)
		print(x_test.shape, y_test.shape)

		new_train_targets = [target for target in new_target_order_no_val if target != test_name]

		assert len(new_train_targets) == 5, 'Unexpected number of datasets'

		X_train = []
		Y_train = []

		train_functions = []
		
		for target_name in new_train_targets:
			x, y = get_x_y_data(targets, target_name)

			X_train.append(x)
			Y_train.append(y)

		assert val_name not in new_train_targets, 'ERROR: validation domain used for training'
		assert test_name not in new_train_targets, 'ERROR: test domain used for training'

if not use_new_benchmarks:
	# check all data is expected sizes and shapes
	for i, x in enumerate(X_train):
		assert x.shape == (20,1), 'X_train['+str(i)+'].shape != (20, 1)'

	assert x_val.shape == (20,1), 'x_val.shape != (20, 1)'
	assert x_val.shape == (20,1), 'x_val.shape != (20, 1)'

	for i, y in enumerate(Y_train):
		assert y.shape == (20,1), 'Y_train['+str(i)+'].shape != (20, 1)'

	assert y_val.shape == (20,1), 'y_val.shape != (20, 1)'
	assert y_val.shape == (20,1), 'y_val.shape != (20, 1)'


# ignore previous validation dataset (later I will remove the previous setup to
# avoid confusion) and create validation datasets for each training dataset.
if not args.use_old_benchmarks:
	print('Validation dataset setup is not implemented for new benchmarks')
	exit()

# make x-values
rng_val = np.random.RandomState(0)

X_val = []
Y_val = []

for i, _ in enumerate(X_train):
	x_val = []
	while len(x_val) < 20:
		while True:
			x_val_value = rng_val.uniform(-1, 1)
			if x_val_value not in X_train[0]:
				break
		x_val.append(x_val_value)
	x_val.sort()
	x_val = np.array(x_val)[:,None]
	X_val.append(x_val)
	Y_val.append(cdff.get_y(x_val, train_functions[i]))

rep = args.rep
exp = args.exp

if args.short_run:
	max_effort = 10
else:
	max_effort = 5*10**10

options['patience'] = args.patience

if args.patience is not None:
	assert args.patience >= 0, '--patience must be non-negative'
	max_effort = float('inf')

rng = np.random.RandomState(args.rep+100*args.exp)

num_inputs_per_dataset = [len(x_test[0])] + [len(x_val[0])] + [len(x[0]) for x in X_train]
num_data_encoder_inputs = max(num_inputs_per_dataset)+1

if args.use_old_primitives:
	primitive_set = ['*', '+', '-']
else:
	primitive_set = ['*', '+', '-', '%', 'sin', 'cos']

terminal_set = ['x'+str(i) for i in range(num_data_encoder_inputs-1)]

if args.use_constants:
	terminal_set.append('#f')

	# if not args.genetic_programming:
	#     terminal_set.append('const_value')

if args.genetic_programming:

	params = {'max_effort': max_effort}
	params['patience'] = args.patience

	if args.simultaneous_targets:

		train_dataset = []

		for i, (x_train, y_train) in enumerate(zip(X_train, Y_train)):

			# format datasets
			train_dataset.append(cdff.combine_x_y(x_train, y_train))

		if args.easy_for_gp:
			test_size = int(len(x_test)/2)
			val_size = len(x_test) - test_size
			test_indices = np.random.choice(len(x_test), test_size, replace=False)
			val_indices = np.array([e for e in range(len(x_test)) if e not in test_indices])

			x_val = x_test[val_indices,:]
			y_val = y_test[val_indices]

			x_test = x_test[test_indices,:]
			y_test = y_test[test_indices]

		val_dataset = []

		for i, (x_val, y_val) in enumerate(zip(X_val, Y_val)):
			val_dataset.append(cdff.combine_x_y(x_val, y_val))
		
		test_dataset = cdff.combine_x_y(x_test, y_test)

		output_path = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(args.exp), 'gp')

		# dataset is the training dataset and validation dataset
		# This is how the datasets are handle for GP.
		# Perhaps I should update this.
		dataset = [train_dataset, val_dataset]
		test_data = test_dataset

		output_file = 'fitness_data_rep' + str(args.rep) + '_train_all_at_once_test_index'+str(args.test_index)+'.csv'

		num_vars = num_data_encoder_inputs-1

		gp = GeneticProgrammingAfpoManyTargetData(rng=rng,
												  pop_size=100,
												  max_gens=600000, # can't set to inf since range(max_gens) used
												  primitive_set=primitive_set,
												  terminal_set=terminal_set,
												  data=dataset,
												  test_data=test_data,
												  prob_mutate=1,
												  prob_xover=0,
												  num_vars=num_vars,
												  mutation_param=2,
												  individual=IndividualManyTargetData,
												  # parameters below
												  **params)

		info = gp.run(rep=args.rep,
					  output_path=output_path,
					  output_file=output_file)

	else:   # for single target

		for i, (x_train, y_train) in enumerate(zip(X_train, Y_train)):

			# format datasets
			train_dataset = cdff.combine_x_y(x_train, y_train)
			val_dataset = cdff.combine_x_y(x_val, y_val)
			test_dataset = cdff.combine_x_y(x_test, y_test)

			output_path = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(args.exp), 'gp')

			# the population from the previous run of
			# genetic programming
			prev_pop = None

			# dataset is the training dataset and validation dataset
			dataset = [train_dataset, val_dataset]
			test_data = test_dataset

			output_file = 'fitness_data_rep' + str(args.rep) + '_train'+str(args.test_index)+'_test_index'+str(args.test_index)+'.csv'

			num_vars = num_data_encoder_inputs-1

			gp = GeneticProgrammingAfpo(rng=rng,
										pop_size=100,
										max_gens=600000,
										primitive_set=primitive_set,
									 	terminal_set=terminal_set,
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

			info = gp.run(rep=args.rep,
						  output_path=output_path,
						  output_file=output_file)

			prev_pop = gp.pop

else:   # for TLC-SR

	save_loc = os.path.join(os.environ['TLCSR_DATA'], 'experiment'+str(args.exp))

	model = TlcsrNetwork(rng=np.random.RandomState(100*args.exp+args.rep),
						 num_data_encoder_inputs=num_data_encoder_inputs, # (xi, ei) -> data encoder
						 primitive_set=primitive_set,
						 terminal_set=terminal_set,
						 use_constants=args.use_constants,
						 timelimit=timelimit,
						 options=options)

	fitter = Tlcsr(rep=rep, exp=exp,
				   model=model,
				   X_train=X_train, Y_train=Y_train,
				   X_val=X_val, Y_val=Y_val,
				   x_test=x_test, y_test=y_test,
				   test_dataset_name='test_index'+str(args.test_index),
				   timelimit=timelimit,
				   simultaneous_targets=args.simultaneous_targets,
				   options=options)

	cmaes_options = {'popsize': 100,
					 'tolfun': 0}  # tolerance in function value

	fitter.fit(max_effort=max_effort,
			   sigma=0.5,
			   cmaes_options=cmaes_options,
			   save_loc=save_loc)
