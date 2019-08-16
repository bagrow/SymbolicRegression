from .IndividualRestricted import IndividualRestricted
from .GeneticProgrammingAfpo import GeneticProgrammingAfpo
from .consts import *

import pandas as pd

import os
import time


class GeneticProgrammingRestricted(GeneticProgrammingAfpo):
    """This class is used to create a population of Individuals
    and evolve them to solve a particular dataset"""

    def __init__(self, rng, pop_size, primitive_set, terminal_set, data,
                 test_data, prob_mutate, prob_xover, num_vars=1,
                 init_max_depth=6, max_depth=17, individual=IndividualRestricted,
                 mutation_param=3, **individual_params):
        """Initialize GeneticProgramming

        Parameters
        ----------
        rng : random number generator
            For example let rng=np.random.RandomState(0)
        pop_size : int
            Number of individuals to put in the population.
        primitive_set : list
            A list of all primitive (operators/functions)
            that may be used in trees.
        terminal_set: list
            A list of all allowed terminals (constants, variables).
        data : np.array
            An np.array of 2D np.arrays. At the top layer, the
            list is split into training and validation datasets.
            Next, into the actual data with output followed by
            each input. That is, a row of data is of the form
            y, x0, x1, ...
        test_data : np.array
            A 2D np.arrays. A row of data is of the form
            y, x0, x1, ...
        num_vars : int (default=1)
            The number of input variables to use. This must be
            specified if more than one input variable is necessary.
        init_max_depth : int (default=6)
            A non-negative integer that limits the depth of the
            trees initially created.
        max_depth : int (default=17)
            A non-negative integer that limits the depth of the
            tree.
        individual : Individual (or superclass)
            The version of the Individual class to use.
        mutation_param : int
            Mutation parameter describing the max depth of subtree
            to be created on mutation.
        """

        self.restrictions = individual_params['restrictions']

        # get min_depth based on self.restrictions
        max_key, max_value = max(self.restrictions.items(), key=lambda x: len(x[0]))

        self.min_depth = len(max_key)

        if max_value in primitive_set:
            self.min_depth += 1

        GeneticProgrammingAfpo.__init__(self, rng, pop_size, primitive_set, terminal_set, data,
                                        test_data, prob_mutate, prob_xover, num_vars,
                                        init_max_depth, max_depth, individual=IndividualRestricted,
                                        mutation_param=mutation_param, **individual_params)

        assert self.min_depth <= self.init_max_depth and self.min_depth <= self.max_depth, 'The init_max_depth or max_depth is not large enough for the given restrictions. These values must be at least '+str(self.min_depth)

        # This is the best individual based
        # on validation error.
        self.best_individual = (float('inf'), None)


    def generate_population_ramped_half_and_half(self, size, init_max_depth):
        """Generate the population using the ramped half and half method.
        Generate equal number of individuals with full and grow method and
        an equal number of each of those with depth of size
        1, 2, 3, ... max_depth.

        Parameters
        ----------
        size : int
            Desired population size
        init_max_depth : int
            Max depth to use when generating trees.
        """

        new_pop = []

        group_size = int(size / (init_max_depth+1-self.min_depth))
        half_group_size = int(group_size / 2)

        # make half the individual with the grow method and the
        # other half with the full method
        # increase max depth as we go
        for d in range(self.min_depth, init_max_depth+1):

            for i in range(half_group_size):

                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='full', max_depth=self.max_depth,
                                               **self.params))

                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='grow', max_depth=self.max_depth,
                                               **self.params))

            # if group size doesn't divide easily make another
            # individual with either grow or full
            if group_size % 2 != 0:

                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='full', max_depth=self.max_depth,
                                               **self.params))

        # if population size is not divisible by group_size
        i = len(new_pop)

        while len(new_pop) < size:

            new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                           depth=(i % init_max_depth)+2, max_depth=self.max_depth,
                                           **self.params))
            i += 1

        return new_pop


    def run_generation(self, gen, num_mut, num_xover):
        """Double the size of the population. Then, remove
        individuals via binary tournament selection.

        Parameters
        ----------
        gen : int
            The current generation.
        num_mut : int
            The number of individuals to generate through
            mutations.
        num_xover : int
            The number of individuals to generate through
            crossover.
        """

        # keep track of best individual
        for p in self.pop:
            if p.validation_fitness < self.best_individual[0]:
                self.best_individual = (p.validation_fitness, p)

        xover_parents = self.rng.choice(self.pop, size=(num_xover, 2))
        mut_parents = self.rng.choice(self.pop, size=num_mut)

        # create one random individual with age 0 and initial max depth of d
        individuals_created = 0
        d = self.rng.randint(1, 3)

        # Make sure d is at least as large as self.min_depth
        d = max(d, self.min_depth)

        # Make sure that d is at most self.max_depth
        # Note that assertion in constructor means that
        # this line will not undo the work of the previous
        # line.
        d = min(d, self.max_depth)

        newborns = [self.Individual(self.rng, self.P, self.T,
                                    num_vars=self.num_vars, age=0, depth=d, max_depth=self.max_depth,
                                    **self.params)]

        xover_count = 0
        mut_count = 0

        for p in self.pop:

            p.age += 1

        # Go through all individuals and edit the population.
        for k in range(self.pop_size):

            # Put the new individual in the population
            if individuals_created == 0:

                self.pop.append(newborns[individuals_created])
                individuals_created += 1

            elif xover_count < num_xover:

                child = self.size_fair_crossover(xover_parents[xover_count])

                self.pop.append(child)
                xover_count += 1

            else:

                # Mutate the individual.
                new_ind = mut_parents[mut_count].mutate(mutation_param=self.mutation_param)

                # Save the parent ID, so we can look at linage later on.
                new_ind.parentID = mut_parents[mut_count].id

                # Put new individual in the population.
                self.pop.append(new_ind)

                mut_count += 1

        # Evaluate the entire population.
        self.compute_fitness()

        # If everything is working, I don't think max_size should ever
        # equal size of front. Thus, it is not necessary to compute the
        # front, which speeds up generations.

        max_size = np.max((self.pop_size, len(self.non_dominated_front)))

        while len(self.pop) > max_size:

            i_loser = None
            num_iterations = 0

            while i_loser is None:

                i_winner, i_loser, winner = self.tournament_selection_multiple_objective(self.rng, self.pop,
                                                                                         replacement=False)

                if num_iterations > 10e6:
                    print('Pareto front is getting too big. Stopping.')
                    exit()

                num_iterations += 1

            del self.pop[i_loser]


    def run(self, rep, output_path=os.environ['GP_DATA'], output_file='fitness_data.csv'):
        """Run the given number of generations with the given parameters.

        Parameters
        ----------
        rep : int
            This number is repetition number.
        output_path : str
            Location to save the data generated. This data includes
            summaries of fintesses and other measurements as well as
            individuals.
        output_file : str
            The filename withtout the path.
        """

        print('output path', output_path)
        print('output file', output_file)

        # compute fitness and save data for generation 0 (random individuals)
        self.compute_fitness()

        info = []
        info.append(self.get_fitness_info(self.pop))
        info[-1].insert(0, 0)

        print(info[-1])

        if self.save_pop_data:

            pop_data = []
            pop_data.extend(self.get_pop_data(0))

        try:
            os.makedirs(output_path)

        except FileExistsError:
            pass

        header = ['Generation', 'Front Size',
                  'Minimum Error', 'Average Error', 'Median Error', 'Maximum Error',
                  'Minimum Objective 2', 'Average Objective 2', 'Median Objective 2', 'Maximum Objective 2',
                  'Minimum Size', 'Average Size', 'Median Size', 'Maximum Size',
                  'Minimum Depth', 'Average Depth', 'Median Depth', 'Maximum Depth',
                  'Minimum Validation Error', 'Average Validation Error', 'Median Validation Error',
                  'Maximum Validation Error']

        if self.save_pop_data:

            pop_data_header = ['Generation', 'Index', 'Root Mean Squared Error', 'Age', 'Equation']
            df_pop = pd.DataFrame(pop_data)
            df_pop.to_csv(os.path.join(output_path, 'pop_data_rep'+str(rep)+'.csv'),
                          header=pop_data_header, index=None)
            info = []

        num_xover = int(self.prob_xover * population_size)

        if num_xover % 2 == 1:
            num_xover -= 1

        num_mut = population_size-num_xover

        # for a fixed number of generations
        for i in range(1, max_generations+1):

            # Do all the generation stuff --- mutate, evaluate...
            self.run_generation(i, num_mut=num_mut, num_xover=num_xover)

            info.append(self.get_fitness_info(self.pop))
            info[-1].insert(0, i)

            print(info[-1])

            if self.save_pop_data:

                pop_data.extend(self.get_pop_data(i))

                # save to the file in chunks
                if i % 1000 == 0:
                    df_pop = pd.DataFrame(pop_data)
                    df_pop.to_csv(os.path.join(output_path, 'pop_data_rep'+str(rep)+'.csv'),
                                  header=None,
                                  index=None,
                                  mode='a')
                    pop_data = []

            # Stop, if ran out of time, but still save stuff.
            if time.time() - self.start_time > self.timeout:
                break

        # Save generational data
        df = pd.DataFrame(info)
        df.to_csv(os.path.join(output_path, output_file), index=None, header=header)

        # Save additional data for the last generation.
        self.save_final_error(os.path.join(output_path, 'fitness_data_rep'+str(rep)+'_final'))

        # Save best individual based on validation error

        lisp = self.best_individual[1].get_lisp_string(actual_lisp=True)

        self.best_individual[1].evaluate_test_points(self.test_data)

        best_data = [lisp,
                     self.best_individual[1].fitness[0],
                     self.best_individual[0],
                     self.best_individual[1].testing_fitness]

        df_best = pd.DataFrame([best_data])
        df_best.to_csv(os.path.join(output_path, 'best_data_rep'+str(rep)+'.csv'),
                       index=False,
                       header=['s-expression', 'Training Error', 'Validation Error', 'Testing Error'])


        # save remaining pop data
        # This is necessary in case the total number
        # of generations is not divisible by 1000.
        if self.save_pop_data:

            df_pop = pd.DataFrame(pop_data)
            df_pop.to_csv(os.path.join(output_path, 'pop_data_rep'+str(rep)+'.csv'),
                          header=None,
                          index=None,
                          mode='a')


        return info
