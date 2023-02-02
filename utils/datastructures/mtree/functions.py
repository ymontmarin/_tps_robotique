import random

from utils.datastructures.mtree.heap_queue import HeapQueue


def random_promotion(data_objects, distance_function):
    """
    Randomly chooses two objects to be promoted.
    """
    data_objects = list(data_objects)
    return random.sample(data_objects, 2)


def balanced_partition(
    promoted_data1, promoted_data2, data_objects, distance_function
):
    partition1 = set()
    partition2 = set()

    queue1 = HeapQueue(
        data_objects,
        key=lambda data: distance_function(data, promoted_data1)
    )
    queue2 = HeapQueue(
        data_objects,
        key=lambda data: distance_function(data, promoted_data2)
    )

    while queue1 or queue2:
        while queue1:
            data = queue1.pop()
            if data not in partition2:
                partition1.add(data)
                break

        while queue2:
            data = queue2.pop()
            if data not in partition1:
                partition2.add(data)
                break

    return partition1, partition2


def make_split_function(promotion_function, partition_function):
    """
    Creates a splitting function.
    The parameters must be callable objects:
        - promotion_function(data_objects, distance_function)
            Must return two objects chosen from the data_objects argument.
        - partition_function(promoted_data1, promoted_data2, data_objects, distance_function)
            Must return a sequence with two iterable objects containing a partition
            of the data_objects. The promoted_data1 and promoted_data2 arguments
            should be used as partitioning criteria and must be contained on the
            corresponding iterable subsets.
    """
    def split_function(data_objects, distance_function):
        promoted_data1, promoted_data2 = promotion_function(
            data_objects, distance_function
        )
        partition1, partition2 = partition_function(
            promoted_data1, promoted_data2, data_objects, distance_function
        )

        return promoted_data1, partition1, promoted_data2, partition2
    return split_function


def make_cached_distance_function(distance_function):
    cache = {}

    def cached_distance_function(data1, data2):
        try:
            distance = cache[data1][data2]
        except KeyError:
            distance = distance_function(data1, data2)

            if data1 in cache:
                cache[data1][data2] = distance
            else:
                cache[data1] = {data2: distance}

            if data2 in cache:
                cache[data2][data1] = distance
            else:
                cache[data2] = {data1: distance}

        return distance

    cached_distance_function.cache = cache

    return cached_distance_function
