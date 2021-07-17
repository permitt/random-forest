import json

import numpy as np
import pandas as pd
import random
import math

from random_forest.forest import Forest

encoder_dict = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3,
                'small': 0, 'med': 1, 'big': 2,
                '2': 0, '3': 1, '4': 2, '5more': 3, 'more': 2,
                'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3
                }


def encode(column):
    return column.apply(lambda val: 1 if column.name == 3 and val == 4 else encoder_dict[val])


def load_dataset(path="./data/car_data.csv"):
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    target_values = [0, 1, 2, 3]
    target_attribute = 'car_evaluation'

    df = pd.read_csv(path, header=None)
    df = df.apply(lambda x: encode(x))
    df.to_csv('./data/car_encoded.csv', index=False, header=False)
    df = df.to_numpy()

    return df, attributes, target_attribute, target_values


class Dataset:

    def __init__(self, data, target_attribute, target_values, attributes=[]):
        self.data = data
        self.entropy = None
        self.target_attribute = target_attribute
        self.target_values = target_values
        self.attributes = attributes

    def do_bagging(self, samples_num=None, no_features=None):

        samples_total = len(self.data[:, 0])  # length of first column == number of samples
        attributes_len = len(self.attributes)
        new_ds = []
        no_features = int(math.sqrt(attributes_len)) if no_features is None else no_features
        samples_num = samples_total // 5 if samples_num is None else samples_num
        attribute_indices = random.sample(range(0, attributes_len), no_features)
        new_attributes = []

        for i in range(samples_num):
            index = random.randint(0, samples_total - 1)
            sample = []
            new_attributes = []

            for k in attribute_indices:
                sample.append(self.data[index, k])
                new_attributes.append(self.attributes[k])

            sample.append(self.data[index, attributes_len])
            new_ds.append(sample)

        dataset_bagged = Dataset(np.array(new_ds), self.target_attribute, self.target_values, new_attributes)

        return dataset_bagged
 
        # bagging treba da vrati random feature i random sampleove, dje moze cak da
        # izabere isti vise puta

    def is_pure(self):
        first_label = self.data[:, -1][0]
        for data in self.data[:, -1]:
            if data != first_label:
                return False

        return True

    def get_pure_label(self):
        return self.data[:, -1][0]

    def split(self, split_on_attribute):
        attribute_index = self.attributes.index(split_on_attribute)
        splitted_datasets = {}

        for index, val in enumerate(self.data[:, attribute_index]):
            new_sample = np.delete(self.data[index], attribute_index)
            if val not in splitted_datasets:
                splitted_datasets[val] = [new_sample]
            else:
                splitted_datasets[val].append(new_sample)

        # kreiraj novi DS
        new_attributes = [x for x in self.attributes if x != split_on_attribute]
        for key, value in splitted_datasets.items():
            value = np.array(value)
            splitted_datasets[key] = Dataset(value, self.target_attribute, self.target_values, new_attributes)

        return splitted_datasets

    def calc_entropy(self):

        if self.entropy:
            return self.entropy

        target = self.data[:, -1]
        total_num = len(target)
        p_values = [np.count_nonzero(target == val) / total_num for val in self.target_values]
        self.entropy = -np.sum([val * np.log2(val + 1) for val in p_values])

        return self.entropy

    def calc_gain_ratio(self, split_on_attribute):

        entropy_before = self.calc_entropy()
        splitted_dict = self.split(split_on_attribute)

        splitted_datasets = splitted_dict.values()

        no_samples_total = len(self.data[:, 0])

        entropy_after = np.sum([data.calc_entropy() for data in splitted_datasets])

        split_info = np.sum(
            [len(dataset.data[:, 0]) / no_samples_total * np.log2(
                len(dataset.data[:, 0]) / no_samples_total) for dataset in
             splitted_datasets]) + 0.000000000001

        information_gain = entropy_before - entropy_after

        gain_ratio = information_gain / split_info
        return gain_ratio, splitted_dict


def writeTreeToFile():
    dataset_array, attributes, target_attribute, target_values = load_dataset()
    dataset = Dataset(dataset_array, target_attribute, target_values, attributes)


    root_node = {}
    forest = Forest(50, dataset, attributes)
    forest.fit()

    for ind, tree in enumerate(forest.trees[:3]):
        root_node["root-"+str(ind)] = {"decision_attribute": tree.root_node.decision_attribute, "value":str(-13), "children":[]}
        for index, child_root in enumerate(tree.root_node.children):
            node = {"decision_attribute": child_root.decision_attribute or str(child_root.label), "value": str(child_root.value) or str(child_root.label), "children": []}
            root_node["root-"+str(ind)]["children"].append(node)
            for index_leaf, child_node in enumerate(child_root.children):
                node_leaf = {"decision_attribute": child_node.decision_attribute or str(child_node.label), "value": str(child_node.value) or str(child_node.label)}
                root_node["root-"+str(ind)]["children"][index]["children"].append(node_leaf)

    print(root_node)
    with open('./data/forest.json', 'w') as f:
        json.dump(root_node, f)
