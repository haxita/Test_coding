import hashlib

class Node(object):
    def __init__(self, parent, state, action, player, depth):
        self.parent = parent
        self.state = state
        self.action = action
        self.player = player
        self.depth = depth

    def key(self):
        # if state is composed of other stuff (dict, set, ...)
        # make it a tuple containing hashable datatypes
        # (this is supposed to be overridden by subclasses)
        return tuple(self.state) + (self.player, )

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if type(self) == type(other):
            return self.key() == other.key()
        raise ValueError('cannot simply compare two different node types')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Node(id:{}, parent:{}, state:{}, action:{}, player:{}, depth:{})'.format(
            id(self),
            id(self.parent),
            self.state,
            self.action,
            self.player,
            self.depth
        )

    def get_move_sequence(self):
        current = self
        reverse_sequence = []
        while current.parent is not None:
            reverse_sequence.append((current.player, current.action))
            current = current.parent
        return list(reversed(reverse_sequence))

    def get_move_sequence_hash(self, nodes_expanded):
        move_sequence = self.get_move_sequence()
        move_sequence_as_str = ';'.join(map(str, move_sequence)) + "|" + str(nodes_expanded)
        move_sequence_hash = hashlib.sha256(move_sequence_as_str.encode('UTF-8')).hexdigest()
        return move_sequence_hash

class Game(object):
    def get_number_of_expanded_nodes(self):
        raise NotImplementedError()

    def get_start_node(self):
        raise NotImplementedError()

    def winner(self, node):
        raise NotImplementedError()

    def successors(self, node):
        raise NotImplementedError()

    def get_max_player(self):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    def get_move_sequence(self, end: Node):
        if end is None:
            return list()
        return end.get_move_sequence()

    def get_move_sequence_hash(self, end: Node):
        if end is None:
            return ''
        return end.get_move_sequence_hash(self.get_number_of_expanded_nodes())

    @staticmethod
    def from_json(jsonstring):
        raise NotImplementedError()

    @staticmethod
    def get_minimum_problem_size():
        raise NotImplementedError()
