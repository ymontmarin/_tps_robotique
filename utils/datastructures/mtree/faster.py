from collections import namedtuple

import utils.datastructures.mtree.functions as functions
from utils.datastructures.mtree.heap_queue import HeapQueue


_INFINITY = float("inf")
_ItemWithDistances = namedtuple(
    '_ItemWithDistances', 'item, distance, min_distance'
)


class _RootNodeReplacement(Exception):
    def __init__(self, new_root):
        super(_RootNodeReplacement, self).__init__(new_root)
        self.new_root = new_root


class _SplitNodeReplacement(Exception):
    def __init__(self, new_nodes):
        super(_SplitNodeReplacement, self).__init__(new_nodes)
        self.new_nodes = new_nodes


class _NodeUnderCapacity(Exception):
    pass


class _IndexItem(object):
    def __init__(self, data):
        self.data = data
        self.radius = 0
        self.distance_to_parent = None

    def _check(self, mtree):
        self._check_data()
        self._check_radius()
        self._check_distance_to_parent()
        return 1

    def _check_data(self):
        assert self.data is not None

    def _check_radius(self):
        assert self.radius is not None
        assert self.radius >= 0

    def _check_distance_to_parent(self):
        assert not isinstance(self, _RootNodeTrait), self
        assert self.distance_to_parent is not None
        assert self.distance_to_parent >= 0


class _Node(_IndexItem):

    def __init__(self, data):
        super(_Node, self).__init__(data)
        self.children = {}

    def add_data(self, data, distance, mtree):
        self.do_add_data(data, distance, mtree)
        self.check_max_capacity(mtree)

    def check_max_capacity(self, mtree):
        if len(self.children) > mtree.max_node_capacity:
            data_objects = frozenset(self.children.keys())
            cached_distance_function = functions.make_cached_distance_function(
                mtree.distance_function
            )

            (promoted_data1, partition1,
             promoted_data2, partition2) = mtree.split_function(
                data_objects, cached_distance_function
            )

            split_node_repl_class = self.get_split_node_replacement_class()
            new_nodes = []
            for promoted_data, partition in [(promoted_data1, partition1),
                                             (promoted_data2, partition2)]:
                new_node = split_node_repl_class(promoted_data)
                for data in partition:
                    child = self.children[data]
                    distance = cached_distance_function(promoted_data, data)
                    new_node.add_child(child, distance, mtree)
                new_nodes.append(new_node)

            raise _SplitNodeReplacement(new_nodes)

    def remove_data(self, data, distance, mtree):
        self.do_remove_data(data, distance, mtree)
        if len(self.children) < self.get_min_capacity(mtree):
            raise _NodeUnderCapacity()

    def update_metrics(self, child, distance):
        child.distance_to_parent = distance
        self.update_radius(child)

    def update_radius(self, child):
        self.radius = max(self.radius, child.distance_to_parent + child.radius)

    def _check(self, mtree):
        super(_Node, self)._check(mtree)
        self._check_min_capacity(mtree)
        self._check_max_capacity(mtree)

        child_height = None
        for data, child in self.children.items():
            assert child.data == data
            self._check_child_class(child)
            self._check_child_metrics(child, mtree)

            height = child._check(mtree)
            if child_height is None:
                child_height = height
            else:
                assert child_height == height

        return child_height + 1

    def _check_max_capacity(self, mtree):
        assert len(self.children) <= mtree.max_node_capacity

    def _check_child_class(self, child):
        expected_class = self._get_expected_child_class()
        assert isinstance(child, expected_class)

    def _check_child_metrics(self, child, mtree):
        dist = mtree.distance_function(child.data, self.data)
        assert child.distance_to_parent == dist, (
            child.data,
            self.data,
            child.distance_to_parent,
            dist,
            abs(child.distance_to_parent - dist)
        )
        assert child.distance_to_parent + child.radius <= self.radius


class _RootNodeTrait(_Node):

    def _check_distance_to_parent(self):
        assert self.distance_to_parent is None


class _NonRootNodeTrait(_Node):

    def get_min_capacity(self, mtree):
        return mtree.min_node_capacity

    def _check_min_capacity(self, mtree):
        assert len(self.children) >= mtree.min_node_capacity


class _LeafNodeTrait(_Node):

    def do_add_data(self, data, distance, mtree):
        entry = _Entry(data)
        assert data not in self.children
        self.children[data] = entry
        assert data in self.children
        self.update_metrics(entry, distance)

    def add_child(self, child, distance, mtree):
        assert child.data not in self.children
        self.children[child.data] = child
        assert child.data in self.children
        self.update_metrics(child, distance)

    @staticmethod
    def get_split_node_replacement_class():
        return _LeafNode

    def do_remove_data(self, data, distance, mtree):
        del self.children[data]

    @staticmethod
    def _get_expected_child_class():
        return _Entry


class _NonLeafNodeTrait(_Node):

    CandidateChild = namedtuple('CandidateChild', 'node, distance, metric')

    def do_add_data(self, data, distance, mtree):

        min_radius_increase_needed = self.CandidateChild(None, None, _INFINITY)
        nearest_distance = self.CandidateChild(None, None, _INFINITY)

        distances = mtree.distance_function(
            data, [child.data for child in self.children.values()]
        )

        for distance, child in zip(distances, self.children.values()):
            if distance > child.radius:
                radius_increase = distance - child.radius
                if radius_increase < min_radius_increase_needed.metric:
                    min_radius_increase_needed = self.CandidateChild(
                        child, distance, radius_increase
                    )
            else:
                if distance < nearest_distance.metric:
                    nearest_distance = self.CandidateChild(
                        child, distance, distance
                    )

        if nearest_distance.node is not None:
            chosen = nearest_distance
        else:
            chosen = min_radius_increase_needed

        child = chosen.node
        try:
            child.add_data(data, chosen.distance, mtree)
        except _SplitNodeReplacement as e:
            assert len(e.new_nodes) == 2
            # Replace current child with new nodes
            del self.children[child.data]
            distances = mtree.distance_function(
                data, [new_child.data for new_child in e.new_nodes]
            )
            for distance, new_child in zip(distances, e.new_nodes):
                self.add_child(new_child, distance, mtree)
        else:
            self.update_radius(child)

    def add_child(self, new_child, distance, mtree):
        new_children = [(new_child, distance)]
        while new_children:
            new_child, distance = new_children.pop()

            if new_child.data not in self.children:
                self.children[new_child.data] = new_child
                self.update_metrics(new_child, distance)
            else:
                existing_child = self.children[new_child.data]
                assert existing_child.data == new_child.data

                # Transfer the _children_ of the new_child to the existing_child
                for grandchild in new_child.children.values():
                    existing_child.add_child(grandchild, grandchild.distance_to_parent, mtree)

                try:
                    existing_child.check_max_capacity(mtree)
                except _SplitNodeReplacement as e:
                    del self.children[new_child.data]
                    distances = mtree.distance_function(
                        self.data, [new_node.data for new_node in e.new_nodes]
                    )
                    for distance, new_node in zip(distances, e.new_nodes):
                        new_children.append((new_node, distance))

    @staticmethod
    def get_split_node_replacement_class():
        return _InternalNode

    def do_remove_data(self, data, distance, mtree):
        for child in self.children.values():
            if abs(distance - child.distance_to_parent) <= child.radius:   # TODO: confirm
                distance_to_child = mtree.distance_function(data, child.data)
                if distance_to_child <= child.radius:
                    try:
                        child.remove_data(data, distance_to_child, mtree)
                    except KeyError:
                        # If KeyError was raised, then the data was not found in the child
                        pass
                    except _NodeUnderCapacity:
                        expanded_child = self.balance_children(child, mtree)
                        self.update_radius(expanded_child)
                        return
                    else:
                        self.update_radius(child)
                        return
        raise KeyError()

    def balance_children(self, the_child, mtree):
        # Tries to find another_child which can donate a grandchild to the_child.

        nearest_donor = None
        distance_nearest_donor = _INFINITY

        nearest_merge_candidate = None
        distance_nearest_merge_candidate = _INFINITY

        distances = mtree.distance_function(
            the_child.data,
            [another_child.data for another_child in (child for child in self.children.values() if child is not the_child)]
        )

        for distance, another_child in zip(distances, (child for child in self.children.values() if child is not the_child)):
            if len(another_child.children) > another_child.get_min_capacity(mtree):
                if distance < distance_nearest_donor:
                    distance_nearest_donor = distance
                    nearest_donor = another_child
            else:
                if distance < distance_nearest_merge_candidate:
                    distance_nearest_merge_candidate = distance
                    nearest_merge_candidate = another_child

        if nearest_donor is None:
            # Merge
            distances = mtree.distance_function(
                nearest_merge_candidate.data,
                [grandchild.data for grandchild in the_child.children.values()]

            )
            for distance, grandchild in zip(distances, the_child.children.values()):
                nearest_merge_candidate.add_child(grandchild, distance, mtree)

            del self.children[the_child.data]
            return nearest_merge_candidate
        else:
            # Donate
            # Look for the nearest grandchild
            nearest_grandchild_distance = _INFINITY
            distances = mtree.distance_function(
                the_child.data,
                [grandchild.data for grandchild in nearest_donor.children.values()]
            )
            for distance, grandchild in zip(distances, nearest_donor.children.values()):
                if distance < nearest_grandchild_distance:
                    nearest_grandchild_distance = distance
                    nearest_grandchild = grandchild

            del nearest_donor.children[nearest_grandchild.data]
            the_child.add_child(nearest_grandchild, nearest_grandchild_distance, mtree)
            return the_child

    @staticmethod
    def _get_expected_child_class():
        return (_InternalNode, _LeafNode)


class _RootLeafNode(_RootNodeTrait, _LeafNodeTrait):

    def remove_data(self, data, distance, mtree):
        try:
            super(_RootLeafNode, self).remove_data(data, distance, mtree)
        except _NodeUnderCapacity:
            assert len(self.children) == 0
            raise _RootNodeReplacement(None)

    @staticmethod
    def get_min_capacity(mtree):
        return 1

    def _check_min_capacity(self, mtree):
        assert len(self.children) >= 1


class _RootNode(_RootNodeTrait, _NonLeafNodeTrait):

    def remove_data(self, data, distance, mtree):
        try:
            super(_RootNode, self).remove_data(data, distance, mtree)
        except _NodeUnderCapacity:
            # Promote the only child to root
            (the_child,) = self.children.values()
            if isinstance(the_child, _InternalNode):
                new_root_class = _RootNode
            else:
                assert isinstance(the_child, _LeafNode)
                new_root_class = _RootLeafNode

            new_root = new_root_class(the_child.data)
            distances = mtree.distance_function(
                new_root.data,
                [grandchild.data for grandchild in the_child.children.values()]
            )

            for distance, grandchild in zip(distances, the_child.children.values()):
                new_root.add_child(grandchild, distance, mtree)

            raise _RootNodeReplacement(new_root)

    @staticmethod
    def get_min_capacity(mtree):
        return 2

    def _check_min_capacity(self, mtree):
        assert len(self.children) >= 2


class _InternalNode(_NonRootNodeTrait, _NonLeafNodeTrait):
    pass


class _LeafNode(_NonRootNodeTrait, _LeafNodeTrait):
    pass


class _Entry(_IndexItem):
    pass


class MTree(object):
    """
    A data structure for indexing objects based on their proximity.

    The data objects must be any hashable object and the support functions
    (distance and split functions) must understand them.

    See http://en.wikipedia.org/wiki/M-tree
    """

    ResultItem = namedtuple('ResultItem', 'data, distance')

    def __init__(
        self,
        distance_function,
        min_node_capacity=20,
        max_node_capacity=None,
        split_function=functions.make_split_function(
            functions.random_promotion, functions.balanced_partition
        )
    ):
        """
        Creates an M-Tree.

        The argument min_node_capacity must be at least 2.
        The argument max_node_capacity should be at least 2*min_node_capacity-1.
        The optional argument distance_function must be a function which calculates
        the distance between two data objects.
        The optional argument split_function must be a function which chooses two
        data objects and then partitions the set of data into two subsets
        according to the chosen objects. Its arguments are the set of data objects
        and the distance_function. Must return a sequence with the following four values:
            - First chosen data object.
            - Subset with at least [min_node_capacity] objects based on the first
                chosen data object. Must contain the first chosen data object.
            - Second chosen data object.
            - Subset with at least [min_node_capacity] objects based on the second
                chosen data object. Must contain the second chosen data object.
        """
        if min_node_capacity < 2:
            raise ValueError("min_node_capacity must be at least 2")
        if max_node_capacity is None:
            max_node_capacity = 2 * min_node_capacity - 1
        if max_node_capacity <= min_node_capacity:
            raise ValueError("max_node_capacity must be greater than min_node_capacity")

        self.min_node_capacity = min_node_capacity
        self.max_node_capacity = max_node_capacity
        self.distance_function = distance_function
        self.split_function = split_function
        self.root = None

    def add(self, data):
        """
        Adds and indexes an object.

        The object must not currently already be indexed!
        """
        if self.root is None:
            self.root = _RootLeafNode(data)
            self.root.add_data(data, 0, self)
        else:
            distance = self.distance_function(data, self.root.data)
            try:
                self.root.add_data(data, distance, self)
            except _SplitNodeReplacement as e:
                assert len(e.new_nodes) == 2
                self.root = _RootNode(self.root.data)
                distances = self.distance_function(
                    self.root.data,
                    [new_node.data for new_node in e.new_nodes]
                )
                for distance, new_node in zip(distances, e.new_nodes):
                    self.root.add_child(new_node, distance, self)

    add_point = add

    def remove(self, data):
        """
        Removes an object from the index.
        """
        if self.root is None:
            raise KeyError()

        distance_to_root = self.distance_function(data, self.root.data)
        try:
            self.root.remove_data(data, distance_to_root, self)
        except _RootNodeReplacement as e:
            self.root = e.new_root

    def get_nearest(self, query_data, range=_INFINITY, limit=_INFINITY):
        """
        Returns an iterator on the indexed data nearest to the query_data. The
        returned items are tuples containing the data and its distance to the
        query_data, in increasing distance order. The results can be limited by
        the range (maximum distance from the query_data) and limit arguments.
        """
        if self.root is None:
            # No indexed data!
            return

        distance = self.distance_function(query_data, self.root.data)
        min_distance = max(distance - self.root.radius, 0)

        pending_queue = HeapQueue(
            content=[_ItemWithDistances(item=self.root, distance=distance, min_distance=min_distance)],
            key=lambda iwd: iwd.min_distance,
        )

        nearest_queue = HeapQueue(key=lambda iwd: iwd.distance)

        yielded_count = 0

        while pending_queue:
            pending = pending_queue.pop()

            node = pending.item
            assert isinstance(node, _Node)

            distances = self.distance_function(
                query_data,
                [child.data for child in node.children.values()]
            )
            for child_distance, child in zip(distances, node.children.values()):
                if abs(pending.distance - child.distance_to_parent) - child.radius <= range:
                    child_min_distance = max(child_distance - child.radius, 0)
                    if child_min_distance <= range:
                        iwd = _ItemWithDistances(item=child, distance=child_distance, min_distance=child_min_distance)
                        if isinstance(child, _Entry):
                            nearest_queue.push(iwd)
                        else:
                            pending_queue.push(iwd)

            # Tries to yield known results so far
            if pending_queue:
                next_pending = pending_queue.head()
                next_pending_min_distance = next_pending.min_distance
            else:
                next_pending_min_distance = _INFINITY

            while nearest_queue:
                next_nearest = nearest_queue.head()
                assert isinstance(next_nearest, _ItemWithDistances)
                if next_nearest.distance <= next_pending_min_distance:
                    _ = nearest_queue.pop()
                    assert _ is next_nearest

                    yield self.ResultItem(data=next_nearest.item.data, distance=next_nearest.distance)
                    yielded_count += 1
                    if yielded_count >= limit:
                        # Limit reached
                        return
                else:
                    break

    def nearest_neighbour(self, point):
        return next(self.get_nearest(point, limit=1))

    def _check(self):
        if self.root is not None:
            self.root._check(self)
