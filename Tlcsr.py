import numpy as np
import pandas as pd
import cma

import os


class Tlcsr():

	def __init__(self, exp, rep, model, X_train, Y_train,
				 x_val, y_val, x_test, y_test,
				 test_dataset_name, simultaneous_targets,
				 timelimit, options=None):
		"""Initialize class with a model to train
		and data to train it on. We would like to 
		approximate the function f, where f(x) = y.

		Parameters
		----------
		model : TlcsrNetwork
			The model that will be trained by CMA-ES
		X_train : list of np.array
			The input data to {f_1, ..., f_m}.
			In each element of the list,
			each column is one variable.
		Y_train : list of np.array
			The output data to {f_1, ..., f_m}.
			In each element of the list,
			is a column array.
		x_val : np.array
			The input validation data to some
			function (not {f_1, ..., f_m}).
			Each column is one variable.
		y_val : np.array
			The output validation data to some
			function (not {f_1, ..., f_m}).
			Shape is column array.
		x_test : np.array
			The input test data to some
			function (not {f_1, ..., f_m}).
			Each column is one variable.
		y_test : np.array
			The output test data to some
			function (not {f_1, ..., f_m}).
			Shape is column array.
		test_dataset_name : str
			The name of the test data. This will appear
			in the name of the output files.
		simultaneous_targets : bool
			Train on multiple targets and validation
			and test on different targets.
		timelimit : int
			The maximum number of equation rewrites.
		options : dict
			Additional options.
			use_k-epxressions : bool
			save_pop_summary : bool
		"""

		self.simultaneous_targets = simultaneous_targets

		if options is None:
			self.options = {}

		else:
			self.options = options

		if 'save_pop_summary' not in options:
			self.options['save_pop_summary'] = False

		if 'use_k-expressions' not in options:
			self.options['use_k-expressions'] = False

		if 'save_lisp_summary' not in options:
			self.options['save_lisp_summary'] = False

		self.rep = rep
		self.exp = exp

		self.seed = 100*self.exp + self.rep + 1

		self.rng = np.random.RandomState(self.seed)
		np.random.seed(self.seed) # Need this for cma to be consistent

		self.model = model

		# Pick 5 so that the length of this
		# is correct. This choice will be
		# randomized before each generation.
		self.X_train = X_train[:5]
		self.Y_train = Y_train[:5]

		self.all_train_datasets = [(x, y) for x, y in zip(X_train, Y_train)]

		# The following is import when using
		# multiple target functions but not
		# training simultaneously.
		self.target_index = 0
		self.y_train = Y_train[self.target_index]
		self.x_train = X_train[self.target_index]
		self.target_index += 1

		self.x_val = x_val
		self.y_val = y_val

		self.x_test = x_test
		self.y_test = y_test

		self.best = {'val error': float('inf')}

		self.test_dataset_name = test_dataset_name
		self.timelimit = timelimit


	@staticmethod
	def get_effort_per_generation(n, popsize, mu):
		"""Computational effor of CMA-ES

		Parameters
		----------
		n : int
			Number of weights
		popsize : int
			The number of individuals in the population
		mu : int
			The number of individuals used to make the average
		"""

		# eq: 5, 9, y, 24, 30, c_1, c_mu, 31, 37
		return (2*n) + 2*(mu-1)*n + (n+1) + (2*n+7) + (2+4*mu+4*n**2+n) + (2) + (3) + (6 + 5*n**2) + (2*n+5)


	def fit(self, max_effort, save_loc, sigma=0.5, cmaes_options=None):
		"""Fit the model to the dataset.

		Parameters
		----------
		max_effort : int
			The maximum number of mathematical operations
			allowed to use during fitting. Once this number
			is reached, save the model and stop fitting.
		sigma : float
			Paramter for CMA-ES: determines the initial standard
			deviation of normal distribution to create population.
		cmaes_options : dict (default={})
			Options to be passed to CMAEvolutionaryStrategy.
		"""

		if cmaes_options is None:
			cmaes_options = {}

		cmaes_options['seed'] = self.seed

		# Initialize weights
		num_weights = self.model.get_num_weights()
		weights = np.random.uniform(-1, 1, size=num_weights)

		es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

		cma_ops_per_gen = self.get_effort_per_generation(n=num_weights,
														popsize=es.popsize,
														mu=es.sp.weights.mu)

		os.makedirs(save_loc, exist_ok=True)

		# initialize f_hat to the zero function
		initial_f_hat = lambda x: 0*x[0]

		if self.options['use_k-expressions']:
			initial_f_hat_seq = ['-', 'x0', 'x0']

		else:
			initial_f_hat_seq = ['-', 'x0', 'x0', 'STOP']

		gen = 0

		best_individual_data = [['Generation',
								 'Target Index',
								 'Train Error Average',
								 # Train errors by themselves go here
								 'Validation Error',
								 'Test Error',
								 'Number of Unique Validation Errors',
								 'Number of Validation Errors',
								 'Number of Floating Point Operations']]

		if len(self.X_train) > 1 and self.simultaneous_targets:

			for i, _ in enumerate(self.X_train):

				best_individual_data[0].insert(3+i, 'Train Error '+str(i))


		effort_checkpoint = 0

		if self.options['save_lisp_summary']:
			summary_data_order = self.model.primitive_set + self.model.terminal_set + ['unique subtrees under -', '- simplified']
			summary_data_table = [['generation', 'NN index', 'dataset type', 'num rewrites'] + summary_data_order]

		while max_effort >= self.model.effort:

			if self.options['save_pop_summary']:
				pop_data_summary = [['Generation', 'individual index', *['train'+str(i) for i in range(len(self.X_train))], 'validation']]

			while not es.stop():

				solutions = es.ask()

				# Average RMSE of the training
				# datasets. These are the values
				# we tell CMA-ES about.
				fitnesses = []

				# The RMSE on the validation dataset.
				# This are the values used to pick
				# the best weight config.
				val_errors = []

				# Both of these lists count the
				# number of equations produced by
				# each solution (weight config) by
				# counting the number of errors produced.
				num_unique_errors = []
				num_errors = []

				# List of error for each target
				# for all weight configuration in solutions.
				individual_errors = []

				# pick datasets for this generation
				if 5 != len(self.all_train_datasets):
					indices = self.rng.choice(len(self.all_train_datasets), 5, replace=False)

					self.X_train = [x for x, y in self.all_train_datasets[indices]]
					self.Y_train = [y for x, y in self.all_train_datasets[indices]]


				# evaluate solutions (weights)
				for nn_index, w in enumerate(solutions):

					if self.options['save_pop_summary']:
						row_pop_data_summary = [gen, nn_index]

					self.model.set_weights(weights=w)

					if self.simultaneous_targets:

						# List of error for each target
						# for a particular weight configuration.
						individual_error_row = []

						for i, (x, y) in enumerate(zip(self.X_train, self.Y_train)):

							output = self.model.evaluate(x, y,
												 		 initial_f_hat, initial_f_hat_seq)

							individual_error_row.append(output['error_best'])

							if self.options['save_pop_summary']:
								row_pop_data_summary.append(output['error_best'])

							if self.options['save_lisp_summary']:
								for j, counts in enumerate(self.model.summary_data):
									row = [gen, nn_index, 'train'+str(i), j] + [counts[key] for key in summary_data_order]
									summary_data_table.append(row)

						individual_errors.append(individual_error_row)

					else:

						output = self.model.evaluate(self.x_train, self.y_train,
													 initial_f_hat, initial_f_hat_seq,
													 return_equation=True,
													 return_decoded_list=True)


					if len(self.X_train) > 1 and self.simultaneous_targets:
						fitnesses.append(np.mean(individual_error_row))

					else:
						fitnesses.append(output['error_best'])

					val_output = self.model.evaluate(self.x_val, self.y_val,
													 initial_f_hat, initial_f_hat_seq,
													 return_equation=True,
													 return_decoded_list=True,
													 return_errors=True)

					val_errors.append(val_output['error_best'])
					num_unique_errors.append(len(np.unique(val_output['errors'])))
					num_errors.append(len(val_output['errors']))

					if self.options['save_pop_summary']:
						row_pop_data_summary.append(val_output['error_best'])
						pop_data_summary.append(row_pop_data_summary)

					if self.options['save_lisp_summary']:
						for j, counts in enumerate(self.model.summary_data):
							row = [gen, nn_index, 'validation', j] + [counts[key] for key in summary_data_order]
							summary_data_table.append(row)

				# Let CMA-ES update the weights based on
				# the fitnesses computed during evaluation.
				es.tell(solutions, fitnesses)
				es.disp()

				# Keep track of best (lowest fitness)
				# individual
				best_index = np.argmin(val_errors)

				if val_errors[best_index] <= self.best['val error']:
					self.model.set_weights(weights=solutions[best_index])

					self.best['val error'] = val_errors[best_index]
					self.best['weights'] = solutions[best_index]
					self.best['network'] = self.model.network
					self.best['fitness'] = fitnesses[best_index]
					self.best['num unique errors'] = num_unique_errors[best_index]
					self.best['num errors'] = num_errors[best_index]
					self.best['training errors'] = fitnesses[best_index] if not self.simultaneous_targets else individual_errors[best_index]

					print('new best', self.best['val error'])

				# update number of operations
				self.model.effort += cma_ops_per_gen

				# Don't count effort to evaluated test
				# because test is only used for analysis.
				effort_before_test = self.model.effort

				# save best individuals during training
				self.model.set_weights(weights=self.best['weights'])

				test_output = self.model.evaluate(self.x_test, self.y_test,
											 	  initial_f_hat, initial_f_hat_seq,
											 	  return_equation=True,
											 	  return_decoded_list=True,
											 	  return_errors=True)


				if self.options['save_lisp_summary']:
					for j, counts in enumerate(self.model.summary_data):
						row = [gen, 'best', 'test', j] + [counts[key] for key in summary_data_order]
						summary_data_table.append(row)

				# Don't count effort to evaluated test
				# because test is only used for analysis.
				self.model.effort = effort_before_test

				row = [gen,
					   self.target_index,
					   self.best['fitness'],
					   self.best['val error'],
					   test_output['error_best'],
					   self.best['num unique errors'],
					   self.best['num errors'],
					   self.model.effort]

				if len(self.X_train) > 1 and self.simultaneous_targets:

					for i, _ in enumerate(self.X_train):

						row.insert(3+i, self.best['training errors'][i])

				best_individual_data.append(row)

				print('total effort', self.model.effort)

				# check if max number of operations have occured
				if max_effort < self.model.effort:
					break

				if not self.simultaneous_targets:
					if self.model.effort - effort_checkpoint > max_effort/len(self.Y_train):

						self.y_train = self.Y_train[self.target_index]
						self.x_train = self.X_train[self.target_index]

						self.target_index += 1
						
						effort_checkpoint += max_effort/len(self.Y_train)
						print('changing target')

						# At this point, it would be ideal if self.model.effort is
						# equal to effort_checkpoint to allow roughly equal effort
						# between targets. So, we cheat and reselt self.model.effort
						self.model.effort = effort_checkpoint

				gen += 1

			es.result_pretty()

			if max_effort >= self.model.effort:
				# restart cma-es with different seed
				weights = self.rng.uniform(-1, 1, size=num_weights)
				cmaes_options['seed'] += 10**7	 # increase by large number, so won't accidentally reuse
				es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

		# Save data about the fitting
		df = pd.DataFrame(best_individual_data[1:], columns=best_individual_data[0])
		df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'), index=False)

		if self.options['save_pop_summary']:
			df = pd.DataFrame(pop_data_summary[1:], columns=pop_data_summary[0])
			df.to_csv(os.path.join(save_loc, 'pop_data_summary_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'), index=False)

		self.best['network'].save_weights(os.path.join(save_loc, 'best_ind_model_weights_rep'+str(self.rep)+'_'+self.test_dataset_name+'.h5'))

		if self.options['save_lisp_summary']:
			df = pd.DataFrame(summary_data_table[1:], columns=summary_data_table[0])
			df.to_csv(os.path.join(save_loc, 'lisp_summary_data_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'), index=False)


if __name__ == '__main__':

	from TlcsrNetwork import TlcsrNetwork

	model = TlcsrNetwork(rng=np.random.RandomState(0),
						 num_data_encoder_inputs=2,
						 primitive_set=['*', '+', '-'],
						 terminal_set=['x0'],
						 options={'use_k-expressions': True,
						 		  'head_length': 5})

	x = np.linspace(-1, 1, 20)[None, :]
	f = lambda x: x[0]**4 + x[0]**3
	y = f(x)

	f_val = lambda x: x[0]**5
	y_val = f_val(x)
	
	f_test = lambda x: x[0]
	y_test = f_test(x)

	rep = 1
	exp = 100

	fitter = Tlcsr(rep=rep, exp=exp, model=model,
				   X_train=[x], Y_train=[y],
				   x_val=x, y_val=y_val,
				   x_test=x, y_test=y_test,
				   simultaneous_targets=True,
				   timelimit=100,
				   test_dataset_name='test')

	cmaes_options = {'popsize': 100,
					 'tolfun': 0}  # toleration in function value

	fitter.fit(max_effort=10**9, sigma=0.5, cmaes_options=cmaes_options)
