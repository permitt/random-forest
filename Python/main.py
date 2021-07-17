from random_forest.dataset import load_dataset, Dataset, writeTreeToFile
from random_forest.forest import Forest
import time
import cProfile
import json

from random_forest.forest_parallel import ForestParallel


def strong_scaling(num_processes):
    dataset_array, attributes, target_attribute, target_values = load_dataset()
    dataset = Dataset(dataset_array, target_attribute, target_values, attributes)

    results = {}

    for val in num_processes:
        start = time.time()
        rf_parallel = ForestParallel(700, dataset, attributes, val)
        rf_parallel.fit()
        results[val] = time.time() - start


    writeTreeToFile()
    with open('./data/strong_scaling.txt', 'w') as f:
       json.dump(results, f)



def weak_scaling(num_of_trees, num_processes):
    dataset_array, attributes, target_attribute, target_values = load_dataset()
    dataset = Dataset(dataset_array, target_attribute, target_values, attributes)

    results = {}

    for index, val  in enumerate(num_processes):
        start = time.time()
        rf_parallel = ForestParallel(num_of_trees[index], dataset, attributes, val)
        rf_parallel.fit()

        results[str((val, num_of_trees[index]))] = time.time() - start

    print(results)

    with open('./data/weak_scaling.txt', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    start = time.time()
    num_of_trees = 50

    num_processes = range(1, 16)
    num_trees = range(100, 3200, 200)

    strong_scaling(num_processes)
    #weak_scaling(num_trees, num_processes)



    # dataset_array, attributes, target_attribute, target_values = load_dataset()
    # dataset = Dataset(dataset_array, target_attribute, target_values, attributes)
    #
    # rf = Forest(num_of_trees, dataset, attributes)
    # #cProfile.run('rf.fit()')
    # rf.fit()
    #
    # sample_1 = dataset.data[1600, :]  # random sample
    #
    # res = rf.predict(sample_1)
    # #print("RESENJE TEST ", sample_1[-1], ' == ', res)
    # print("Sequential time elapsed: ", time.time() - start)
    #
    #
    # start = time.time()
    #
    # rf_parallel = ForestParallel(num_of_trees, dataset, attributes, 2)
    #
    # rf_parallel.fit()
    #
    # sample_1 = dataset.data[1600, :]  # random sample
    #
    # #res = rf_parallel.predict(sample_1)
    # #print("A RESENJE JE OVO BILO ", sample_1[-1], ' == ', res)
    # print("Parallel time elapsed: ", time.time() - start)
    #
    #
    # print(rf.trees_timing)
    # print(rf_parallel.trees_timing)
    #
    # with open('./data/python_sequential.txt', 'w') as f:
    #     [f.write(str(val) + ', ') for val in rf.trees_timing]
    #
    # with open('./data/python_parallel.txt', 'w') as f:
    #     [f.write(str(val) + ', ') for val in rf_parallel.trees_timing]