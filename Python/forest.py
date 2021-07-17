from random_forest.tree import Tree
import math
import time


class Forest:

    def __init__(self, trees_num, dataset, attributes):
        self.trees_num = trees_num
        self.trees = []
        self.no_features = int(math.sqrt(len(attributes) - 1))
        self.dataset = dataset

        # now we do bagging approach
        for i in range(self.trees_num):
            ds = dataset.do_bagging(samples_num=400, no_features=3)
            tree = Tree(ds)
            self.trees.append(tree)

    def fit(self):

        for tree in self.trees:
            tree.fit()

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
