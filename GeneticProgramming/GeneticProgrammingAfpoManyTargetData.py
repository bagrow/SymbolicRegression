from .GeneticProgrammingAfpo import GeneticProgrammingAfpo

import pandas as pd

import os
import time

class GeneticProgrammingAfpoManyTargetData(GeneticProgrammingAfpo):

    def evaluate_individual(self, ind, data):
        """This function is different from original
        because the number of datasets is adjusted
        """

        # rememeber data = [train, val]
        # so we need to adjust train not data...
        indices = self.rng.choice(len(data[0]), 5, replace=False)

        ind.evaluate_individual_error([[data[0][i] for i in indices],data[1]])

        # Age
        ind.fitness[1] = ind.age

        # Number of Nodes
        if self.params['AFSPO']:
            ind.fitness[2] = ind.get_tree_size()


    def run(self, rep, output_path=os.environ['GP_DATA'],
            output_file='fitness_data.csv'):
        """Run the given number of generations with the given parameters.

        ATERATION: Option to stop based on validation error.

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
        self.update_best_individual()

        info = []
        info.append(self.get_fitness_info(self.pop))
        info[-1].insert(0, 0)

        print(info[-1])

        best_data = []

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

        df_best_ind = pd.DataFrame([], columns=['Generation',
                                                's-expression',
                                                'Training Error',
                                                'Validation Error',
                                                'Testing Error',
                                                'CPU Time',
                                                'Computation'])

        df_best_ind.to_csv(os.path.join(output_path, 'best_'+output_file),
                            index=None)

        if self.save_pop_data:

            pop_data_header = ['Generation', 'Index', 'Root Mean Squared Error', 'Age', 'Equation']
            df_pop = pd.DataFrame(pop_data)
            df_pop.to_csv(os.path.join(output_path, 'pop_data_rep'+str(rep)+'.csv'),
                          header=pop_data_header, index=None)
            info = []

        num_xover = int(self.prob_xover * self.pop_size)

        if num_xover % 2 == 1:
            num_xover -= 1

        num_mut = self.pop_size-num_xover

        # for a fixed number of generations
        for i in range(1, self.max_gens+1):

            # Do all the generation stuff --- mutate, evaluate...
            self.run_generation(i, rep=rep, output_path=output_path, num_mut=num_mut, num_xover=num_xover)

            # keep track of best individual
            self.update_best_individual()

            info.append(self.get_fitness_info(self.pop))
            info[-1].insert(0, i)

            print(info[-1])

            # Save best individual based on validation error
            lisp = self.best_individual.get_lisp_string(actual_lisp=True)

            self.best_individual.evaluate_test_points(self.test_data)

            best_data.append([i, lisp,
                              self.best_individual.fitness[0],
                              self.best_individual.validation_fitness,
                              self.best_individual.testing_fitness,
                              time.process_time()-self.start_time,
                              self.effort])

            if i % 1000 == 0:

                df_best = pd.DataFrame(best_data)
                df_best.to_csv(os.path.join(output_path, 'best_'+output_file),
                               index=False,
                               header=None,
                               mode='a')

                best_data = []

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

            if self.params['patience'] is not None:

                # Stop if validation error has stopped improving
                if not hasattr(self, 'prev_best_val'):
                    self.prev_best_val = self.best_individual.validation_fitness
                    self.gen_of_prev_best_val = i
                else:
                    if self.prev_best_val > self.best_individual.validation_fitness:
                        self.prev_best_val = self.best_individual.validation_fitness
                        self.gen_of_prev_best_val = i
                    else:
                        if (i-self.gen_of_prev_best_val) >= self.params['patience']:
                            break

            # Stop, if ran out of time, but still save stuff.
            if time.process_time() - self.start_time > self.timeout or self.max_effort < self.effort:
                break

        # Save generational data
        df = pd.DataFrame(info)
        df.to_csv(os.path.join(output_path, output_file), index=None, header=header)

        # Save additional data for the last generation.
        self.save_final_error(os.path.join(output_path, 'fitness_data_rep'+str(rep)+'_final'))

        df_best = pd.DataFrame(best_data)
        df_best.to_csv(os.path.join(output_path, 'best_'+output_file),
                       index=False,
                       header=None,
                       mode='a')

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