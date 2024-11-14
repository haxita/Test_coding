from collections import deque


class Queue(object):
    def __init__(self):
        self.d = deque()

    def put(self, v):
        self.d.append(v)

    def get(self):
        return self.d.popleft()

    def has_elements(self):
        return len(self.d) > 0

    def __iter__(self):
        return iter(self.d)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('Queue [' + ','.join((str(item) for item in self.d)) + ']')

    def __len__(self):
        return len(self.d)
