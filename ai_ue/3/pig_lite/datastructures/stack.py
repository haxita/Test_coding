from collections import deque


class Stack(object):
    def __init__(self):
        self.d = deque()

    def put(self, v):
        self.d.append(v)

    def get(self):
        return self.d.pop()

    def has_elements(self):
        return len(self.d) > 0

    def __iter__(self):
        return iter(self.d)

    def __repr__(self):
        return ('Stack [' + ','.join((str(item) for item in self.d)) + ']')
