import numpy as np
import pandas as pd
import cma

import os


class CmaesTrainsNn():

	def __init__(self, exp, rep, model, x_train, Y_train,
				 x_val, y_val, x_test, y_test,
				 test_dataset_name,
				 timelimit, options=None):
		"""Initialize class with a model to train
		and data to train it on. We would like to 
		approximate the function f, where f(x) = y.

		Parameters
		----------
		model : Seq2Seq
			The model that will be trained by CMA-ES
		x_train : np.array
			The input data to f.
		Y_train : np.array
			The output data to f.
		x_val : np.array
			The input validation data to f.
		y_val : np.array
			The output validation data to f.
		x_test : np.array
			The input test data to f.
		y_test : np.array
			The output test data to f.
		test_dataset_name : str
			The name of the test data. This will appear
			in the name of the output files.
		"""

		if options is None:
			self.options = {'use_k-expressions': False}

		else:
			self.options = options

		self.rep = rep
		self.exp = exp

		self.seed = 100*self.exp + self.rep + 1

		self.rng = np.random.RandomState(self.seed)
		
		self.model = model

		self.x_train = x_train
		self.Y_train = Y_train
		self.target_index = 0
		self.y_train = Y_train[self.target_index]
		self.target_index += 1

		self.x_val = x_val
		self.y_val = y_val

		self.x_test = x_test
		self.y_test = y_test

		self.best = (float('inf'), None, None)

		self.test_dataset_name = test_dataset_name
		self.timelimit = timelimit


	@staticmethod
	def get_FLoPs_per_generation(n, popsize, mu):
		"""
		Parameters
		----------
		n : int
			Number of weights
		popsize : int
			The number of individuals in the population
		mu : int
			The number of individuals used to make the average?
		"""

		# eq: 5, 9, y, 24, 30, c_1, c_mu, 31, 37
		return (2*n) + 2*(mu-1)*n + (n+1) + (2*n+7) + (2+4*mu+4*n**2+n) + (2) + (3) + (6 + 5*n**2) + (2*n+5)


	def fit(self, max_FLoPs, sigma=0.5, cmaes_options=None):
		"""Fit the model to the dataset.

		Parameters
		----------
		max_FLoPs : int
			The maximum number of floating point operations
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

		num_weights = self.model.get_num_weights(self.model.model)

		# Initialize weights
		weights = np.random.uniform(-1, 1, size=num_weights)

		es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

		cma_ops_per_gen = self.get_FLoPs_per_generation(n=num_weights,
														popsize=es.popsize,
														mu=es.sp.weights.mu)

		save_loc = os.path.join(os.environ['EE_DATA'], 'experiment'+str(self.exp))

		os.makedirs(save_loc, exist_ok=True)

		# initialize f_hat to the zero function
		initial_f_hat = lambda x: 0*x[0]

		if self.options['use_k-expressions']:
			initial_f_hat_seq = ['START', '-', 'x0', 'x0'] + ['x0']*(self.model.head_length+self.model.tail_length-4)

		else:
			initial_f_hat_seq = ['START', '-', 'x0', 'x0', 'STOP']

		gen = 0

		best_individual_data = [['Generation', 'Train Error Sum', 'Validation Error', 'Test Error', 'Number of Floating Point Operations']]
		FLoPs_checkpoint = 0

		while max_FLoPs >= self.model.FLoPs:

			pop_data_summary = []

			while not es.stop():

				solutions = es.ask()

				fitnesses = []
				val_fitnesses = []

				# evaluate solutions (weights)
				for w in solutions:

					self.model.set_weights(model=self.model.model, weights=w)

					output = self.model.evaluate(self.x_train, self.y_train,
												 initial_f_hat, initial_f_hat_seq,
												 return_equation=True,
												 return_decoded_list=True)


					fitnesses.append(output['fitness_sum'])

					val_output = self.model.evaluate(self.x_val, self.y_val,
												 initial_f_hat, initial_f_hat_seq,
												 return_equation=True,
												 return_decoded_list=True)

					val_fitnesses.append(val_output['fitness'])

					# if output['decoded_list'] is not None:
					# 	print('final equation', ' '.join(output['decoded_list']))

				# Let ES update the weights based on
				# the fitness computed during evaluation.
				es.tell(solutions, fitnesses)
				es.disp()

				# Keep track of best (lowest fitness)
				# individual
				best_index = np.argmin(val_fitnesses)

				if val_fitnesses[best_index] <= self.best[0]:
					self.model.set_weights(weights=solutions[best_index], model=self.model.model)
					self.best = (val_fitnesses[best_index], solutions[best_index], self.model.model, fitnesses[best_index])
					print('new best', self.best[0])

				gen += 1

				# update number of operations
				self.model.FLoPs += cma_ops_per_gen

				# save best individuals during training
				self.model.set_weights(weights=self.best[1], model=self.model.model)

				output = self.model.evaluate(self.x_test, self.y_test,
											 initial_f_hat, initial_f_hat_seq,
											 return_equation=True,
											 return_decoded_list=True)

				best_individual_data.append([gen,
											 self.best[3],
											 self.best[0],
											 output['fitness'],
											 self.model.FLoPs])

				print('total compute', self.model.FLoPs)

				# check if max number of computations have occured
				if max_FLoPs < self.model.FLoPs:
					break

				if self.model.FLoPs - FLoPs_checkpoint > max_FLoPs/len(self.Y_train):

					self.y_train = self.Y_train[self.target_index]

					self.target_index += 1
					
					FLoPs_checkpoint += max_FLoPs/len(self.Y_train)
					print('changing target')

			es.result_pretty()

			if max_FLoPs >= self.model.FLoPs:
				# restart cma-es with different seed
				weights = self.rng.uniform(-1, 1, size=num_weights)
				cmaes_options['seed'] += 10**7
				es = cma.CMAEvolutionStrategy(weights, sigma, cmaes_options)

		# Save data about the fitting
		df = pd.DataFrame(best_individual_data[1:], columns=best_individual_data[0])
		df.to_csv(os.path.join(save_loc, 'best_ind_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'), index=False)

		df = pd.DataFrame(pop_data_summary, columns=['Generation', 'Mean Training Error', 'Mean Validation Error'])
		df.to_csv(os.path.join(save_loc, 'pop_data_summary_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'), index=False)

		self.best[2].save(os.path.join(save_loc, 'best_ind_model_rep'+str(self.rep)+'_'+self.test_dataset_name+'.csv'))

if __name__ == '__main__':

	from seq2seq_first import seq2seq

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
