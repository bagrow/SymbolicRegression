from .GeneticProgramming import GeneticProgramming
from .Individual import Individual

import numpy as np

import os


class GeneticProgrammingAfpo(GeneticProgramming):
    """This class inherits GeneticProgramming.
    It adds and alters some functionality. This class
    is an implementation of Age-Fitness Pareto Optimization
    (AFPO).

    Schmidt M., Lipson H. (2011) Age-Fitness Pareto Optimization.
    In: Riolo R., McConaghy T., Vladislavleva E. (eds) Genetic
    Programming Theory and Practice VIII. Genetic and Evolutionary
    Computation, vol 8. Springer, New York, NY"""


    def __init__(self, rng, pop_size, max_gens, primitive_set, terminal_set, data,
                 test_data, prob_mutate, prob_xover, num_vars=1,
                 init_max_depth=6, max_depth=17, individual=Individual,
                 mutation_param=3, **individual_params):

        GeneticProgramming.__init__(self, rng=rng, pop_size=pop_size, max_gens=max_gens,
                                    primitive_set=primitive_set, terminal_set=terminal_set, data=data,
                                    test_data=test_data, prob_mutate=prob_mutate, prob_xover=prob_xover,
                                    num_vars=num_vars, init_max_depth=init_max_depth, max_depth=max_depth,
                                    individual=individual, mutation_param=mutation_param, **individual_params)

        if 'AFSPO' not in self.params:
            self.params['AFSPO'] = False


    def evaluate_individual(self, ind, data):
        """Evaluate the individual fitness and update
        the second fitness objective with the current
        age of the individual.

        Parameters
        ----------
        ind : Individual
            Individual whose fitness is to be calculated.
        data : np.array
            An np.array of 2D np.arrays. At the top layer, the
            list is split into training and validation datasets.
            Next, into the actual data with output followed by
            each input. That is, a row of data is of the form
            y, x0, x1, ...
        """

        ind.evaluate_individual_error(data)

        # Age
        ind.fitness[1] = ind.age

        # Number of Nodes
        if self.params['AFSPO']:
            ind.fitness[2] = ind.get_tree_size()


    def size_fair_crossover(self, parents):
        """Crossover parents (list of 2 nodes) by selection the first
        crossover point normally. Then, ensure a final tree (only one)
        that has depth less than max depth. Child receives max age
        of parents.

        Parameters
        ----------
        parents : iterable (of Individuals)
            These are the two parents to crossover."""

        ind = GeneticProgramming.size_fair_crossover(self, parents)

        ind.age = max((self.non_dominated_front[parents[0]].age, self.non_dominated_front[parents[1]].age))

        return ind


    def run_generation(self, gen, rep, output_path, num_mut, num_xover):
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

            elif p.validation_fitness == self.best_individual[1].get_tree_size():

                if p.get_tree_size() < self.best_individual[1].get_tree_size():
                    self.best_individual = (p.validation_fitness, p)

        xover_parents = self.rng.choice(self.pop, size=(num_xover, 2))
        mut_parents = self.rng.choice(self.pop, size=num_mut)

        # create one random individual with age 0 and initial max depth of d
        individuals_created = 0
        d = self.rng.randint(3, 6)

        # Make sure that d is at most self.max_depth
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

                # if at least one descendant, get its children
                if self.descendants_of_given_individual:
                    if xover_parents[xover_count][0].get_lisp_string() in self.descendants_of_given_individual or xover_parents[xover_count][1].get_lisp_string() in self.descendants_of_given_individual:
                        self.descendants_of_given_individual.append(new_ind.get_lisp_string())

                self.pop.append(child)
                xover_count += 1

            else:

                # Mutate the individual.
                new_ind = mut_parents[mut_count].mutate(mutation_param=self.mutation_param)

                # Save the parent ID, so we can look at linage later on.
                new_ind.parentID = mut_parents[mut_count].id

                # if at least one descendant, get its mutants
                if self.descendants_of_given_individual:
                    if mut_parents[mut_count].get_lisp_string() in self.descendants_of_given_individual:
                        self.descendants_of_given_individual.append(new_ind.get_lisp_string())

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

            # Update descendant list by removing individuals that no longer exist
            if self.descendants_of_given_individual:

                lisp = self.pop[i_loser].get_lisp_string()

                try:

                    self.descendants_of_given_individual.remove(lisp)

                    # This is the first generation that there
                    # no descendants since this list will never
                    # grow if it becomes empty.
                    if not self.descendants_of_given_individual:
                        
                        with open(os.path.join(output_path, 'generation_no_more_descendants_rep'+str(rep)+'.txt'), 'w') as f:
                            f.write(str(gen))

                except ValueError:
                    pass

            del self.pop[i_loser]


    def get_pop_data(self, gen):
        """Get data for each individual in the population.

        Parameters
        ----------
        gen : int
            The generation number.

        Returns
        -------
        A list of lists each row is of the form
        [gen, individual_index, error, age, equation]
        """

        return [[gen, i, p.fitness[0], p.age, p.get_lisp_string()]
                for i, p in enumerate(self.pop)]
