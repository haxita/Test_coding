import hashlib

class Node(object):
    def __init__(self, parent, state, action, cost, depth):
        self.parent = parent
        self.state = state
        self.action = action
        self.cost = cost
        self.depth = depth

    def key(self):
        # if state is composed of other stuff (dict, set, ...)
        # make it a tuple containing hashable datatypes
        # (this is supposed to be overridden by subclasses)
        return self.state

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if type(self) == type(other):
            return self.key() == other.key()
        raise ValueError('cannot simply compare two different node types')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Node(id:{}, parent:{}, state:{}, action:{}, cost:{}, depth:{})'.format(
            id(self),
            id(self.parent),
            self.state,
            self.action,
            self.cost,
            self.depth
        )

    def get_action_sequence(self):
        current = self
        reverse_sequence = []
        while current.parent is not None:
            reverse_sequence.append(current.action)
            current = current.parent
        return list(reversed(reverse_sequence))
    
    def get_action_sequence_hash(self):
        action_sequence = self.get_action_sequence()
        action_sequence_as_str = ','.join(map(str, action_sequence))
        action_sequence_hash = hashlib.sha256(action_sequence_as_str.encode('UTF-8')).hexdigest() # should solution node return hashcode?
        return action_sequence_hash
    
    def pretty_print(self):
        print(f"state {self.state} was reached following the sequence {self.get_action_sequence()} (cost: {self.cost}, depth: {self.depth})")


class Problem(object):
    def get_number_of_expanded_nodes(self):
        raise NotImplementedError()

    def get_start_node(self):
        raise NotImplementedError()

    def get_end_node(self):
        raise NotImplementedError()

    def is_end(self, node):
        raise NotImplementedError()

    def action_cost(self, state, action):
        raise NotImplementedError()

    def successors(self, node):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()
    
    def visualize(self, **kwargs):
        raise NotImplementedError()
    
    def get_action_sequence(self, end: Node):
        if end is None:
            return list()
        return end.get_action_sequence()

    @staticmethod
    def from_json(jsonstring):
        raise NotImplementedError()

    @staticmethod
    def get_minimum_problem_size():
        raise NotImplementedError()
