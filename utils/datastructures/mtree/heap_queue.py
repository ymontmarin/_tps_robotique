from collections import namedtuple


_HeapItem = namedtuple('_HeapItem', 'k, value')

class HeapQueue(object):

    def __init__(self, content=(), key=lambda x: x, max=False):
        if max:
            self.key = lambda x: -key(x)
        else:
            self.key = key
        self._items = [_HeapItem(self.key(value), value) for value in content]
        self.heapify()

    def _items_less_than(self, base, other):
        return self._items[base].k < self._items[other].k

    def _swap_items(self, base, other):
        self._items[base], self._items[other] = self._items[other], self._items[base]

    def _make_heap(self, i):
        smallest = i

        left = 2 * i + 1
        if left < len(self._items) and self._items_less_than(left, smallest):
            smallest = left

        right = 2 * i + 2
        if right < len(self._items) and self._items_less_than(right, smallest):
            smallest = right

        if smallest != i:
            self._swap_items(i, smallest)
            self._make_heap(smallest)

    def heapify(self):
        for i in range(len(self._items) // 2, -1, -1):
            self._make_heap(i)

    def head(self):
        return self._items[0].value

    def push(self, value):
        i = len(self._items)
        new_item = _HeapItem(self.key(value), value)
        self._items.append(new_item)
        while i > 0:
            p = int((i - 1) // 2)
            if self._items_less_than(p, i):
                break
            self._swap_items(i, p)
            i = p

    def pop(self):
        popped = self._items[0].value
        self._items[0] = self._items[-1]
        self._items.pop(-1)
        self._make_heap(0)
        return popped

    def pushpop(self, value):
        k = self.key(value)
        if k <= self._items[0].k:
            return value
        else:
            popped = self._items[0].value
            self._items[0] = _HeapItem(k, value)
            self._make_heap(0)
            return popped

    def __len__(self):
        return len(self._items)

    def extractor(self):
        while self._items:
            yield self.pop()
