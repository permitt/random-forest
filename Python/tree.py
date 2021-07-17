import math


class Tree:

    def __init__(self, dataset):
        self.root_node = None
        self.dataset = dataset
        self.attributes = dataset.attributes
        self.target_attribute = dataset.target_attribute

    def fit(self):
        self.root_node = Node(self.dataset, None)
        self.root_node.fit()

    def predict(self, example):
        return self.root_node.predict(example)


"""
    This is the C4.5 implementation of decision trees. It uses the gain ratio as the metric
    for selecting the best attribute for splitting.
    Algorithm is as follows:
        1. Check if base case is satisfied (pure subset)
            1.1 If pure is True than make the leaf node with that label 
            1.2 If pure is False then proceed to 2.
        2. For every attribute A in allAttributes
            2.1 calculate the gain ratio and return the highest scored attribute S
        3. Make a subtree with root node and decision attribute S
        4. For every subset with values from S check 1.
"""


class Node:

    def __init__(self, dataset, value, is_leaf=False):
        self.dataset = dataset
        self.children = []
        self.decision_attribute = None
        self.value = value
        self.label = None
        self.global_attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    def fit(self):

        if self.dataset.is_pure() or len(self.dataset.attributes) == 0:
            self.label = self.dataset.get_pure_label()
            self.children = []
            return

        attr_score = {}
        max_gr = -math.inf
        max_attr = None

        for attr in self.dataset.attributes:
            attr_gain_ratio, splitted_dict = self.dataset.calc_gain_ratio(attr)
            attr_score[attr] = splitted_dict

            if max_gr < attr_gain_ratio:
                max_attr = attr
                max_gr = attr_gain_ratio

        self.decision_attribute = max_attr
        for value in attr_score[max_attr]:
            child = Node(attr_score[max_attr][value], value)
            self.children.append(child)
            child.fit()

    def predict(self, example):
        if self.is_leaf():
            return self.label

        attribute_index = self.global_attributes.index(self.decision_attribute)

        for node in self.children:
            if example[attribute_index] == node.value:
                return node.predict(example)

    def is_leaf(self):
        return len(self.children) == 0
