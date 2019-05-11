from classes.individual import Individual
from consts import *

import pandas as pd
# import numpy as np

import os
import copy


class GeneticProgramming:


    def __init__(self, rng, pop_size, primitive_set, terminal_set, data, test_data,
                 prob_mutate, prob_xover, num_vars=1, init_max_depth=6, max_depth=17,
                 individual=Individual, mutation_param=3, **individual_params):

        self.max_depth = max_depth
        self.pop_size = pop_size
        self.P = primitive_set
        self.T = terminal_set
        self.data = data
        self.test_data = test_data
        self.rng = rng
        self.prop_mutate = prob_mutate
        self.prob_xover = prob_xover
        self.Individual = individual
        self.params = individual_params
        self.num_vars = num_vars
        self.mutation_param = mutation_param

        assert 0 < self.num_vars <= 10, 'Currently, cannot support more than 10 variables.'

        self.non_dominated_front = {}

        self.pop = self.generate_random_population_ramped_half_and_half(self.pop_size, init_max_depth)

        # mutation parameter stuff
        if self.mutation_param == 7:

            self.mutation_param = self.rng.randint(1, 6)

        elif self.mutation_param == 8:

            self.mutation_param = self.rng.randint(2, 6)

        elif self.mutation_param == 9:

            self.mutation_param = self.rng.randint(4, 5)


    def generate_random_population_ramped_half_and_half(self, size, init_max_depth):

        new_pop = []

        group_size = int(size / (init_max_depth))
        half_group_size = int(group_size / 2)

        # make have the individual with the grow method and the other half with the full method
        # increase max depth as we go
        for d in range(1, init_max_depth+1):

            for i in range(half_group_size):

                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='full', max_depth=self.max_depth,
                                               **self.params))
                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='grow', max_depth=self.max_depth,
                                               **self.params))

            # if group size doesn't divide easily make another individual with either grow or full
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


    def size_fair_crossover(self, parents):
        """Crossover parents (list of 2 nodes) by selection the first
        crossover point normally. Then, ensure a final tree (only one)
        that has depth less than max depth."""

        ind1 = copy.deepcopy(parents[0])
        ind2 = copy.deepcopy(parents[1])

        # do this to make sure that new individual gets new id,
        # even if the same lisp
        ind1 = self.Individual(rng=ind1.rng,
                               primitive_set=ind1.P,
                               terminal_set=ind1.T,
                               num_vars=ind1.num_vars,
                               age=ind1.age,
                               max_depth=ind1.max_depth,
                               node=ind1.tree,
                               node_class=ind1.Node,
                               **self.params)

        # if only root node, return a parent
        if ind1.is_leaf() and ind2.is_leaf():

            return ind1

        list1 = ind1.get_node_list()
        list2 = ind2.get_node_list()

        # pick first crossover point
        c1 = list(self.rng.choice(list1))

        # get depth of c1 node and the subtree connected their
        c1_depth = len(c1)

        # determining max depth of second crossover point
        max_depth = self.max_depth - c1_depth

        # get depth of all subtrees of ind2 with legal depth
        sub_tree_depths = {}

        for i, el in enumerate(list2):

            # get depth of node that defines subtree
            d = len(el)

            # get max depth of labels that start with the label el
            max_branch_depth = np.max([len(l) for l in list2 if l[:len(el)] == el])

            sub_tree_depth = max_branch_depth - d

            if sub_tree_depth > max_depth:
                continue

            if sub_tree_depth in sub_tree_depths:
                sub_tree_depths[sub_tree_depth].append(i)

            else:
                sub_tree_depths[sub_tree_depth] = [i]

        # choose depth
        key = self.rng.choice(list(sub_tree_depths.keys()))

        # choose a specific crossover point
        c2 = list(list2[self.rng.choice(sub_tree_depths[key])])

        # get the subtree, make copy and delete from original
        subtree2 = ind2.select_subtree(c2)

        # swap subtree into other tree
        ind1.set_subtree(subtree2, c1)

        # keep track of ancestry
        parent_ids = (parents[0].id, parents[1].id)

        ind1.parentID = parent_ids

        ind1.age = max([ind1.age, ind2.age])

        return ind1


    def compute_fitness(self):
        """Compute fitness for all individuals. Not the individuals on
        the front so that error is not recalculated."""

        for i, individual in enumerate(self.pop):

            self.evaluate_individual(individual, self.data, False)


    def evaluate_individual(self, ind, data, is_non_dominated):
        """Evaluate the individual fitness and worst neighbors score."""

        ind.evaluate_individual_error(data, is_non_dominated)

        # number of nodes
        ind.fitness[1] = ind.get_tree_size()


    def get_fitness_info(self, group):

        front_size = len(self.non_dominated_front)

        error_list = [individual.fitness[0] for individual in group]

        max_f1 = max(error_list)
        min_f1 = min(error_list)
        avg_f1 = np.mean(error_list)
        median_f1 = np.median(error_list)

        age_list = [individual.fitness[1] for individual in group]

        max_f2 = max(age_list)
        min_f2 = min(age_list)
        avg_f2 = np.mean(age_list)
        median_f2 = np.median(age_list)

        sizes = [ind.get_tree_size() for ind in group]

        max_s = max(sizes)
        min_s = min(sizes)
        avg_s = np.mean(sizes)
        median_s = np.median(sizes)

        depths = [ind.get_depth() for ind in group]

        max_d = max(depths)
        min_d = min(depths)
        avg_d = np.mean(depths)
        median_d = np.median(depths)

        val_error = [individual.validation_fitness for individual in group]

        max_v = max(val_error)
        min_v = min(val_error)
        avg_v = np.mean(val_error)
        median_v = np.median(val_error)

        return [front_size,
                min_f1, avg_f1, median_f1, max_f1,
                min_f2, avg_f2, median_f2, max_f2,
                min_s, avg_s, median_s, max_s,
                min_d, avg_d, median_d, max_d,
                min_v, avg_v, median_v, max_v]


    def get_non_dominated_front(self):
        """Look at the population and determine which individual are non-dominated. Store these individual in a
        dict called non_dominated_front. The keys are the index in the population."""

        # Initialize the non dominated front as a dict for easy comparison
        self.non_dominated_front = {}

        # Check each individual against every other individual. If none dominate than the current individual
        # is on the front.
        for i, ind in enumerate(self.pop):

            for j, ind2 in enumerate(self.pop):

                # This should be the most common.
                if ind2.dominates(ind):

                    break

            else:  # no break (no one dominates ind)

                self.non_dominated_front[i] = copy.deepcopy(ind)
                self.pop[i].parentID = None


    def run_generation(self, gen, num_mut, num_xover):

        # Make a dictionary of front individuals. Keep key as index.
        self.get_non_dominated_front()

        print('size of front', len(self.non_dominated_front))

        xover_parents = self.rng.choice(self.pop, size=(num_xover, 2))
        mut_parents = self.rng.choice(self.pop, size=num_mut)

        xover_count = 0
        mut_count = 0

        inds_to_replace = [k for k, p in enumerate(self.pop) if k not in self.non_dominated_front]

        # Go through all individuals and edit the population.
        for k in inds_to_replace:

            if xover_count < num_xover:

                child = self.size_fair_crossover(xover_parents[xover_count])

                self.pop[k] = child
                xover_count += 1

            else:

                # Mutate the individual.
                new_ind = mut_parents[mut_count].mutate(mutation_param=self.mutation_param)

                # Save the parent ID, so we can look at linage later on.
                new_ind.parentID = mut_parents[mut_count].id

                # Put new individual in the population.
                self.pop[k] = new_ind

                mut_count += 1

        # Evaluate the entire population. Skip error objective for individuals on front.
        self.compute_fitness()


    def save_final_error(self, filename):
        """Save training, validation, and testing error to filename."""

        # Get data.
        fitness_all = [[ind.get_lisp_string(),
                        ind.fitness[0],
                        ind.validation_fitness,
                        ind.evaluate_test_points_fast(self.test_data),
                        ind.fitness[1]] for ind in self.pop]

        # Save data.
        df = pd.DataFrame(fitness_all)

        df.to_csv(filename+'.csv', index=True,
                  header=['Equation',
                          'Training',
                          'Validation',
                          'Testing',
                          'Objective 2 on Training Data'])

        if len(self.non_dominated_front) != 0:

            # Save the data again, but exclude anyone not on the front.
            fitness_front = [[key,
                              self.non_dominated_front[key].get_lisp_string(),
                              self.non_dominated_front[key].fitness[0],
                              self.non_dominated_front[key].validation_fitness,
                              self.non_dominated_front[key].evaluate_test_points_fast(self.test_data),
                              self.non_dominated_front[key].fitness[1]] for key in self.non_dominated_front]

            df = pd.DataFrame(fitness_front)

            df.to_csv(filename+'_front.csv', index=False,
                      header=['Index',
                              'Equation',
                              'Training',
                              'Validation',
                              'Testing',
                              'Objective 2 on Training Data'])


    def run(self, rep, output_path=os.environ['GP_DATA'], output_file='fitness_data.csv'):
        """Run the given number of generations with the given parameters."""

        print('output path', output_path)
        print('output file', output_file)

        # compute fitness and save data for generation 0 (random individuals)
        self.compute_fitness()

        info = []
        info.append(self.get_fitness_info(self.pop))
        info[-1].insert(0, 0)

        print(info[-1])

        if not os.path.exists(output_path):

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

        num_xover = int(self.prob_xover * population_size)

        if num_xover % 2 == 1:

            num_xover -= 1

        if not os.path.exists(output_path):

            os.makedirs(output_path)

        num_mut = population_size-num_xover

        # for a fixed number of generations
        for i in range(1, max_generations+1):

            # Do all the generation stuff --- mutate, evaluated, compute front...
            self.run_generation(i, num_mut=num_mut, num_xover=num_xover)

            info.append(self.get_fitness_info(self.pop))
            info[-1].insert(0, i)

            print(info[-1])

            # if front is the whole population, further progress is not possible, so terminate.
            if len(self.non_dominated_front) == population_size:
                break

        # Save generational data
        df = pd.DataFrame(info)
        df.to_csv(output_path + output_file, index=None, header=header)

        # Save additional data for the last generation.
        self.save_final_error(output_path+'fitness_data_rep'+str(rep)+'_final')

        return info

    # ------------------------------------------------------------ #
    #        Strength Pareto Evolutionary Algorithm (SPEA2)
    # ------------------------------------------------------------ #
    # These functions are not curently used.


    def spea2(self, rng, archive_size, archive, population, number_xo_parents, number_mut_parents):
        """Algorithm description here: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/145755/eth-24689-01.pdf
        Output the nondominated front"""

        # Initialization
        apop = archive + population

        # Fitness Assignment
        distances, fitnesses = self.get_fitnesses_spea2(apop)

        # Environmental Selection
        fitnesses_index_sorted = np.argsort(fitnesses)
        apop_sorted = np.array(apop)[fitnesses_index_sorted]
        fitnesses_sorted = list(np.array(fitnesses)[fitnesses_index_sorted])
        distances_sorted = distances[fitnesses_index_sorted, :]

        new_archive = []

        for i, (indiv, fit) in enumerate(zip(apop_sorted, fitnesses_sorted)):

            if fit < 1:

                indiv.age += 1

                # Update second fitness objective
                self.evaluate_individual(indiv, self.data, is_non_dominated=True)

                new_archive.append(indiv)

            else:

                break

        non_dominated_front = dict(zip(np.array(range(len(apop)))[fitnesses_index_sorted], copy.deepcopy(new_archive)))

        if len(new_archive) == archive_size:

            pass    # done

        elif len(new_archive) < archive_size:

            while len(new_archive) < archive_size:

                apop_sorted[len(new_archive)].age += 1

                # Update second fitness objective
                self.evaluate_individual(apop_sorted[len(new_archive)], self.data, is_non_dominated=True)

                # Add to archive
                new_archive.append(apop_sorted[len(new_archive)])

        else:   # len(new_archive) > archive_size

            while len(new_archive) > archive_size:

                for index, i in enumerate(new_archive):

                    for jndex, j in enumerate(new_archive):

                        if index == jndex:

                            continue

                        elif not self.lessthan(index, jndex, distances_sorted, len(new_archive)):

                            break   # don't remove this one

                    else:   # no-break

                        del new_archive[index]
                        del fitnesses_sorted[index]

                        distances_sorted = list(distances_sorted)
                        del distances_sorted[index]
                        distances_sorted = np.array(distances_sorted)
                        break

        ox_parents = np.vstack(([self.tournament_selection(rng, new_archive, fitnesses_sorted, 2, replacement=True) for _ in range(number_xo_parents)],
                                [self.tournament_selection(rng, new_archive, fitnesses_sorted, 2, replacement=True) for _ in range(number_xo_parents)])).T

        mut_parents = [self.tournament_selection(rng, new_archive, fitnesses_sorted, 2, replacement=True)
                       for _ in range(number_mut_parents)]

        return (new_archive, non_dominated_front, ox_parents, mut_parents)


    def lessthan(self, index, jndex, distances, length_new_archive):

        for k in range(1, length_new_archive):

            if distances[index, k] == distances[jndex, k]:

                continue

            elif distances[index, k] < distances[jndex, k]:

                return True

            else:

                return False

        return True


    def get_strength(self, i, apop):
        """apop is short for archive and population."""

        return len([j for j in apop if i.dominates(j)])


    def get_raw_fitnesses(self, apop):
        """apop is short for archive and population."""

        strengths = [self.get_strength(i, apop) for i in apop]

        return np.array([np.sum([strengths[index] for index, j in enumerate(apop) if j.dominates(i)]) for i in apop])


    def get_desnsity(self, apop):

        distances = [[np.linalg.norm(i.fitness-j.fitness) for j in apop] for i in apop]

        k = np.sqrt(len(apop))
        distances = np.sort(distances, axis=1)

        sigmas = distances[:, int(k)]

        return (distances, 1./(sigmas+2.))


    def get_fitnesses_spea2(self, apop):

        distances, D = self.get_desnsity(apop)
        R = self.get_raw_fitnesses(apop)

        return (distances, R + D)


    def tournament_selection(self, rng, archive, archive_fitnesses, k, replacement=True):

        index1, index2 = rng.choice(len(archive), size=k, replace=replacement)

        if archive_fitnesses[index1] < archive_fitnesses[index2]:

            return archive[index1]

        else:

            return archive[index2]


    def tournament_selection_multiple_objective(self, rng, pop, k, replacement=True):

        index1, index2 = rng.choice(len(pop), size=k, replace=replacement)

        if pop[index1].dominates(pop[index2]):

            return index1, index2, pop[index1]

        elif pop[index2].dominates(pop[index1]):

            return index2, index1, pop[index2]

        else:

            return None, None, None
