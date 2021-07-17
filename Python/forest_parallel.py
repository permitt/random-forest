from random_forest.tree import Tree
from multiprocessing import Pool, cpu_count
import math


class ForestParallel:

    def __init__(self, trees_num, dataset, attributes, num_processes=cpu_count()):
        self.trees_num = trees_num
        self.trees = []
        self.no_features = int(math.sqrt(len(attributes) - 1))
        self.dataset = dataset
        self.num_processes = num_processes

        # now we do bagging approach
        for i in range(self.trees_num):
            ds = dataset.do_bagging(samples_num=400, no_features=3)
            tree = Tree(ds)
            self.trees.append(tree)

    def fit_tree(self, tree):
        tree.fit()

    def fit(self):
        with Pool(processes=self.num_processes) as pool:
            pool.map(self.fit_tree, self.trees)


    def predict(self, sample):

        results = {}

        for key in self.dataset.target_values:
            results[key] = 0

        for tree in self.trees:
            inference = tree.predict(sample)
            if inference is not None:
                results[inference] += 1

        print(results)

        max_score = -math.inf
        max_attr = None
        for key, value in results.items():
            if value >= max_score:
                max_score = value
                max_attr = key

        return max_attr
