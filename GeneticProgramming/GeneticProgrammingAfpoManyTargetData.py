from .GeneticProgrammingAfpo import GeneticProgrammingAfpo

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
