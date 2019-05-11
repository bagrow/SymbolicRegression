from classes.GeneticProgramming import GeneticProgramming
from classes.individual import Individual

import numpy as np


class GeneticProgrammingAfpo(GeneticProgramming):


    def __init__(self, rng, pop_size, primitive_set, terminal_set, data, test_data,
                 prob_mutate, prob_xover, num_vars=1, init_max_depth=6, max_depth=17,
                 individual=Individual, **individual_params):

        GeneticProgramming.__init__(self, rng, pop_size, primitive_set, terminal_set, data, test_data,
                                    prob_mutate, prob_xover, num_vars, init_max_depth, max_depth,
                                    individual, **individual_params)


    def evaluate_individual(self, ind, data, non_dominated_front):
        """Evaluate the individual fitness and worst neighbors score."""

        ind.evaluate_individual_error(data)

        # Age
        ind.fitness[1] = ind.age


    def size_fair_crossover(self, parents):
        """Crossover parents (list of 2 nodes) by selection the first
        crossover point normally. Then, ensure a final tree (only one)
        that has depth less than max depth.

        Child receives max age of parents"""

        ind = GeneticProgramming.size_fair_crossover(self, parents)

        ind.age = max((self.non_dominated_front[parents[0]].age, self.non_dominated_front[parents[1]].age))

        return ind


    def run_generation(self, gen, num_mut, num_xover):
        """Compute front, then mutate and evaluate individuals."""

        xover_parents = self.rng.choice(self.pop, size=(num_xover, 2))
        mut_parents = self.rng.choice(self.pop, size=num_mut)

        # create one random individual with age 0 and initial max depth of d
        individuals_created = 0
        d = self.rng.randint(3, 6)
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

        # Evaluate the entire population. Skip error objective for individuals on front.
        self.compute_fitness()

        # If everything is working, I don't think max_size should ever equal size of front.
        # Thus, it is not necessary to compute the front, which speeds up generations.
        # self.get_non_dominated_front()

        max_size = np.max((self.pop_size, len(self.non_dominated_front)))

        while len(self.pop) > max_size:

            i_loser = None
            num_iterations = 0

            while i_loser is None:

                i_winner, i_loser, winner = self.tournament_selection_multiple_objective(self.rng, self.pop,
                                                                                         k=2, replacement=False)

                if num_iterations > 10e6:
                    print('Pareto front is getting too big. Stopping.')
                    exit()

                num_iterations += 1

            del self.pop[i_loser]


    def get_pop_data(self, gen):
        """Get data for each individual in the population.

        [gen, indiv_index, error, age, equation]"""

        return [[gen, i, p.fitness[0], p.age, p.get_lisp_string()]
                for i, p in enumerate(self.pop)]
