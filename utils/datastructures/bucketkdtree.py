import numpy as np
from cspace_metric.datastructures.tree import NodeBinaryTree


class BucketKDNode(NodeBinaryTree):
    # Adapt to wraparound spaces ?
    def __init__(
        self,
        parent=None,
        points=None,
        dim=3,
        dim_scale=None,
        bucketsize=10
    ):
        # Parent node
        self.parent = parent
        # Hyperparam
        self.bucketsize = bucketsize
        self.dim = dim
        self.dim_scale = dim_scale if dim_scale is not None else np.ones(dim)
        # Node are initially leaf without children and with a bucket
        self.is_leaf = True
        # Points buckets storage_data for leaf node
        self._points = np.zeros((bucketsize, dim))
        self._n_points = 0
        # Value usefull for not leaf node
        self.split_dim = None
        self.split_val = None
        self.left = None
        self.right = None
        # Bounds tracker
        self.lower = np.ones((dim)) * np.inf
        self.upper = - np.ones((dim)) * np.inf
        # Add given points
        self.add_points(points)

    def _update_bounds(self, points):
        # ensure non null points
        if not isinstance(points, np.ndarray):
            if not points:
                return
            points = np.array(points)
        if points.size is 0:
            return

        mini = np.min(points, axis=0)
        maxi = np.max(points, axis=0)
        min_update = mini < self.lower
        self.lower[min_update] = mini[min_update]
        max_update = maxi > self.upper
        self.upper[max_update] = maxi[max_update]

    def add_point(self, p):
        self._update_bounds([p])
        if self.is_leaf:
            # The node is still a leaf, just add to bucket
            # We are sure there is enough space
            self._points[self._n_points] = p
            self._n_points += 1
            if self._n_points == self.bucketsize:
                # It is now full, transform node as non leaf
                self._create_children()
        else:
            # Kd split to add to children
            if p[self.split_dim] <= self.split_val:
                self.left.add_point(p)
            else:
                self.right.add_point(p)

    def add_points(self, points):
        # ensure non null points
        if not isinstance(points, np.ndarray):
            if not points:
                return
            points = np.array(points)
        if points.size is 0:
            return

        if self.is_leaf:
            # We add the maximum we can to the bucket
            n = min(len(points), self.bucketsize - self._n_points)
            batch = points[:n]
            self._points[self._n_points:self._n_points + n] = batch
            self._n_points += n
            self._update_bounds(batch)
            if self._n_points == self.bucketsize:
                # It is full, transform node as non leaf
                self._create_children()
            # Add eventual remaining points
            self.add_points(points[n:])
        else:
            # We add points to child given their position
            self._update_bounds(points)
            infe = points[:, self.split_dim] <= self.split_val
            self.left.add_points(points[infe])
            self.right.add_points(points[~infe])

    def _create_children(self):
        assert self.is_leaf and self.bucketsize == self._n_points
        # The creation must appeaar only for full leaf (after an add)
        # At this points the bounds are the one of the bucket
        ranges = self.upper - self.lower
        split_dim = np.argmax(ranges * self.dim_scale)
        # No more a leaf, create attribute for non leaf
        self.is_leaf = False
        self.split_dim = split_dim
        self.split_val = self.lower[split_dim] + ranges[split_dim] / 2
        self.left = BucketKDNode(
            self,
            dim=self.dim, dim_scale=self.dim_scale, bucketsize=self.bucketsize
        )
        self.right = BucketKDNode(
            self,
            dim=self.dim, dim_scale=self.dim_scale, bucketsize=self.bucketsize
        )
        # Now diffuse bucket points in children and erase local bucket
        self.add_points(self._points)
        self._points = None

    def nearest_neighbour(self, query, dist_to_many, max_dist=None):
        if self.is_leaf:
            dists = dist_to_many(query, self._points[:self._n_points])
            i_min = np.argmin(dists)
            if max_dist is None or dists[i_min] < max_dist:
                return dists[i_min], self._points[i_min]
            return None, None
        else:
            cursor = self
            # Go down to the best leaf
            while not cursor.is_leaf:
                if query[cursor.split_dim] <= cursor.split_val:
                    cursor = cursor.left
                else:
                    cursor = cursor.right
            best_d, best_p = cursor.nearest_neighbour(
                query, dist_to_many, max_dist
            )
            if best_d is not None:
                max_dist = best_d
            # Go up by recursively checking ambiguous split
            while cursor is not self:
                cursor = cursor.parent
                # check ambiguity to non coming child
                # Get nearest in the child if needed
                i = cursor.split_dim
                s = self.dim_scale[i]
                x = query[i]
                d, p = None, None
                if x <= cursor.split_val:
                    # We come from left, check right
                    if s * (cursor.right.lower[i] - x) < max_dist:
                        # There is an  ambiguity, check right
                        d, p = cursor.right.nearest_neighbour(
                            query, dist_to_many, max_dist
                        )
                else:
                    # Same for right
                    if s * (x - cursor.left.upper[i]) < max_dist:
                        d, p = cursor.left.nearest_neighbour(
                            query, dist_to_many, max_dist
                        )
                if d is not None:
                    # We have found something better in ambiguity
                    best_d, best_p = d, p
                    max_dist = best_d

            return best_d, best_p


class SBucketKDNode(NodeBinaryTree):
    """
    Alternative with external storage_data
    """
    def __init__(
        self,
        storage_data,
        parent=None,
        points_idx=None,
        dim=3,
        dim_scale=None,
        bucketsize=10
    ):
        # Store storage_data
        self.storage_data = storage_data
        # Parent node
        self.parent = parent
        # Hyperparam
        self.bucketsize = bucketsize
        self.dim = dim
        self.dim_scale = dim_scale if dim_scale is not None else np.ones(dim)
        # Node are initially leaf without children and with a bucket
        self.is_leaf = True
        # Points buckets storage_data for leaf node
        self._points_idx = np.zeros(bucketsize, dtype=np.intp)
        self._n_points = 0
        # Value usefull for not leaf node
        self.split_dim = None
        self.split_val = None
        self.left = None
        self.right = None
        # Bounds tracker
        self.lower = np.ones(dim, dtype=float) * np.inf
        self.upper = - np.ones(dim, dtype=float) * np.inf
        # Add given points
        self.add_points(points_idx)

    def _update_bounds(self, points_idx):
        # ensure non null points
        if not isinstance(points_idx, np.ndarray):
            if not points_idx:
                return
            points_idx = np.array(points_idx, dtype=np.intp)
        if points_idx.size is 0:
            return
        mini = np.min(self.storage_data[points_idx], axis=0)
        maxi = np.max(self.storage_data[points_idx], axis=0)
        min_update = mini < self.lower
        self.lower[min_update] = mini[min_update]
        max_update = maxi > self.upper
        self.upper[max_update] = maxi[max_update]

    def add_point(self, p_idx):
        self._update_bounds([p_idx])
        if self.is_leaf:
            # The node is still a leaf, just add to bucket
            # We are sure there is enough space
            self._points_idx[self._n_points] = p_idx
            self._n_points += 1
            if self._n_points == self.bucketsize:
                # It is now full, transform node as non leaf
                self._create_children()
        else:
            # Kd split to add to children
            if self.storage_data[p_idx, self.split_dim] <= self.split_val:
                self.left.add_point(p_idx)
            else:
                self.right.add_point(p_idx)

    def add_points(self, points_idx):
        # ensure non null points
        if not isinstance(points_idx, np.ndarray):
            if not points_idx:
                return
            points_idx = np.array(points_idx, dtype=np.intp)
        if points_idx.size is 0:
            return

        if self.is_leaf:
            # We add the maximum we can to the bucket
            n = min(len(points_idx), self.bucketsize - self._n_points)
            batch = points_idx[:n]
            self._points_idx[self._n_points:self._n_points + n] = batch
            self._n_points += n
            self._update_bounds(batch)
            if self._n_points == self.bucketsize:
                # It is full, transform node as non leaf
                self._create_children()
            # Add eventual remaining points
            self.add_points(points_idx[n:])
        else:
            # We add points to child given their position
            self._update_bounds(points_idx)
            infe = self.storage_data[points_idx, self.split_dim] <= self.split_val
            self.left.add_points(points_idx[infe])
            self.right.add_points(points_idx[~infe])

    def _create_children(self):
        assert self.is_leaf and self.bucketsize == self._n_points
        # The creation must appeaar only for full leaf (after an add)
        # At this points the bounds are the one of the bucket
        ranges = self.upper - self.lower
        split_dim = np.argmax(ranges * self.dim_scale)
        # No more a leaf, create attribute for non leaf
        self.is_leaf = False
        self.split_dim = split_dim
        self.split_val = self.lower[split_dim] + ranges[split_dim] / 2
        self.left = SBucketKDNode(
            self.storage_data, self,
            dim=self.dim, dim_scale=self.dim_scale, bucketsize=self.bucketsize
        )
        self.right = SBucketKDNode(
            self.storage_data, self,
            dim=self.dim, dim_scale=self.dim_scale, bucketsize=self.bucketsize
        )
        # Now diffuse bucket points in children and erase local bucket
        self.add_points(self._points_idx)
        self._points_idx = None

    def nearest_neighbour(self, query, dist_to_many, max_dist=None):
        if self.is_leaf:
            dists = dist_to_many(
                query, self.storage_data[self._points_idx[:self._n_points]]
            )
            i_min = np.argmin(dists)
            if max_dist is None or dists[i_min] < max_dist:
                return dists[i_min], self._points_idx[i_min]
            return None, None
        else:
            cursor = self
            # Go down to the best leaf
            while not cursor.is_leaf:
                if query[cursor.split_dim] <= cursor.split_val:
                    cursor = cursor.left
                else:
                    cursor = cursor.right
            best_d, best_p = cursor.nearest_neighbour(
                query, dist_to_many, max_dist
            )
            if best_d is not None:
                max_dist = best_d
            # Go up by recursively checking ambiguous split
            while cursor is not self:
                cursor = cursor.parent
                # check ambiguity to non coming child
                # Get nearest in the child if needed
                i = cursor.split_dim
                s = self.dim_scale[i]
                x = query[i]
                d, p = None, None
                if x <= cursor.split_val:
                    # We come from left, check right
                    if s * (cursor.right.lower[i] - x) < max_dist:
                        # There is an  ambiguity, check right
                        d, p = cursor.right.nearest_neighbour(
                            query, dist_to_many, max_dist
                        )
                else:
                    # Same for right
                    if s * (x - cursor.left.upper[i]) < max_dist:
                        d, p = cursor.left.nearest_neighbour(
                            query, dist_to_many, max_dist
                        )
                if d is not None:
                    # We have found something better in ambiguity
                    best_d, best_p = d, p
                    max_dist = best_d

            return best_d, best_p
