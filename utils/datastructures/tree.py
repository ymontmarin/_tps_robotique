class NodeBinaryTree:
    """
    Abstract tree to implement the classic search
    """
    def __init__(
        self, parent=None, left=None, right=None
    ):
        self.parent = parent
        self.left = left
        self.right = right

    def ascension(self):
        yield self
        if self.parent is not None:
            for e in self.parent.ascension():
                yield e

    def depth_first(self):
        yield self
        if self.left is not None:
            for e in self.left.depth_first():
                yield e
        if self.right is not None:
            for e in self.right.depth_first():
                yield e

    def _wide_first(self, i=0):
        yield self, i
        iter_left = (
            iter(self.left._wide_first(i + 1))
            if self.left is not None else None
        )
        iter_right = (
            iter(self.right._wide_first(i + 1))
            if self.right is not None else None
        )
        i_left, n_left = self.robust_next(iter_left)
        i_right, n_right = self.robust_next(iter_right)
        while not (i_left is None and i_right is None):
            if i_left is not None and (i_right is None or i_left <= i_right):
                yield i_left, n_left
                i_left, n_left = self.robust_next(iter_left)
            else:
                yield i_right, n_right
                i_right, n_right = self.robust_next(iter_right)

    def wide_first(self):
        for _, e in self._wide_first():
            yield e

    @staticmethod
    def robust_next(iterator):
        if iterator is None:
            return (None, None)
        return next(iterator, (None, None))
