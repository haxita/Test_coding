import json
import numpy as np

from copy import deepcopy
from pig_lite.game.base import Node, Game


class TTTNode(Node):
    def key(self):
        return tuple(self.state.flatten().tolist() + [self.player])

    def __repr__(self):
        return '"TTTNode(\nid:{}\nparent:{}\nboard:\n{}\nplayer:\n{}\naction:\n{}\ndepth:{})"'.format(
            id(self),
            id(self.parent),
            # this needs to be printed transposed, so it fits together with
            # how matplotlib's 'imshow' renders images
            self.state.T,
            self.player,
            self.action,
            self.depth
        )

    def pretty_print(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(['tab:blue', 'lightgray', 'tab:orange'])
        print('State of the board:')
        plt.figure(figsize=(2, 2))
        plt.imshow(self.state.T, cmap=cm)
        plt.axis('off')
        plt.show()
        print('Performed moves: {}'.format(self.depth))


class TicTacToe(Game):
    def __init__(self, rng=None, depth=None):
        self.n_expands = 0
        self.play_randomly(rng, depth)

    def play_randomly(self, rng, depth):
        """ Initialises self.start_node to be either empty board, or board at given depth after random playing. """
        empty_board = np.zeros((3, 3), dtype=int)
        start_from_empty = TTTNode(None, empty_board, None, 1, 0)
        if rng is None or depth is None or depth == 0:
            self.start_node = start_from_empty
        else:
            # proceed playing randomly until either 'depth' is reached,
            # or the node is a terminal node
            nodes = []
            successors = [start_from_empty]
            while True:
                index = rng.randint(0, len(successors))
                current = successors[index]

                if current.depth == depth:
                    break

                nodes.append(current)
                terminal, winner = self.outcome(current)
                if terminal:
                    break
                successors = self.successors(current)

                for node in successors:
                    nodes.append(node)

            self.start_node = TTTNode(None, current.state, None, current.player, 0)

    def get_start_node(self):
        """ Returns start node of this Game. """
        return self.start_node

    def outcome(self, node):
        """ Returns tuple stating whether game is finished or not, and winner (or None otherwise). """
        board = node.state
        for player in [-1, 1]:
            # checks rows and columns
            for i in range(3):
                if (board[i, :] == player).all() or (board[:, i] == player).all():
                    return True, player

            # checks diagonals
            if (np.diag(board) == player).all() or (np.diag(np.rot90(board)) == player).all():
                return True, player

        # if board is full, and none of the conditions above are true,
        # nobody has won --- it's a draw
        if (board != 0).all():
            return True, None

        # else, continue
        return False, None

    def get_max_player(self):
        """ Returns identifier of MAX player used in this game. """
        return 1

    def successor(self, node, action):
        """ Performs given action at given game node, and returns successor TTT node. """
        board = node.state
        player = node.player

        next_board = board.copy()
        next_board[action] = player

        if player == 1:
            next_player = -1
        else:
            next_player = 1

        return TTTNode(
            node,
            next_board,
            action,
            next_player,
            node.depth + 1
        )

    def get_number_of_expanded_nodes(self):
        return self.n_expands

    def successors(self, node):
        """ Given a game node, returns all possible successor nodes based on all actions that can be performed. """
        self.n_expands += 1
        terminal, winner = self.outcome(node)

        if terminal:
            return []
        else:
            successor_nodes = []
            # iterate through all possible coordinates (==actions)
            for action in zip(*np.nonzero(node.state == 0)):
                successor_nodes.append(self.successor(node, action))
            return successor_nodes

    def to_json(self):
        """ Converts and stores this TTT game to a JSON file. """
        return json.dumps(dict(
            type=self.__class__.__name__,
            start_state=self.start_node.state.tolist(),
            start_player=self.start_node.player
        ))

    @staticmethod
    def from_json(jsonstring):
        """ Loads given JSON file, and creates game with information. """
        data = json.loads(jsonstring)

        ttt = TicTacToe()
        ttt.start_node = TTTNode(
            None,
            np.array(data['start_state'], dtype=int),
            None,
            data['start_player'],
            0
        )
        return ttt

    @staticmethod
    def from_dict(data):
        """ Creates game with information in given data-dictionary. """
        ttt = TicTacToe()
        ttt.start_node = TTTNode(
            None,
            np.array(data['start_state'], dtype=int),
            None,
            data['start_player'],
            0
        )
        return ttt

    @staticmethod
    def get_minimum_problem_size():
        return 0

    def visualize(self, move_sequence, show_possible=False, tree_name=''):
        game = deepcopy(self)
        nodes = []
        current = game.get_start_node()
        nodes.append(current)
        for player, move in move_sequence:
            if show_possible:
                successors = game.successors(current)
                nodes.extend(successors)
                current = None
                for succ in successors:
                    if succ.action == move:
                        current = succ
                        break
            else:
                current = game.successor(current, move)
                nodes.append(current)

        try:
            self.networkx_plot_game_tree(tree_name, nodes)
        except ImportError:
            print('#' * 30)
            print('#' * 30)
            print('starting position')
            print(self.get_start_node())
            print('#' * 30)
            print('#' * 30)
            print('-' * 30)
            print('sequence of nodes')
            for node in nodes:
                print('-' * 30)
                print(node)
                terminal, winner = game.outcome(node)
                print('terminal {}, winner {}'.format(terminal, winner))

    def networkx_plot_game_tree(self, title, nodes, highlight=None):
        # TODO: this needs some serious refactoring
        # use visitors for styling, for example, instead of cumbersome dicts
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, VPacker, TextArea

        fig, tree_ax = plt.subplots()
        tree_ax.set_title(title)
        G = nx.DiGraph(ordering='out')
        nodes_extra = dict()
        edges_extra = dict()

        def sort_key(node):
            if node.action is None:
                return (-1, -1)
            return node.action

        for node in sorted(nodes, key=sort_key):
            G.add_node(id(node), search_node=node)
            terminal, winner = self.outcome(node)
            nodes_extra[id(node)] = dict(
                board=node.state,
                player=node.player,
                depth=node.depth,
                terminal=terminal,
                winner=winner
            )

        for node in nodes:
            if node.parent is not None:
                edge = id(node.parent), id(node)
                G.add_edge(*edge, parent_node=node.parent)
                edges_extra[edge] = dict(
                    label='{}'.format(node.action),
                    parent_player=node.parent.player
                )

        node_size = 1000
        positions = graphviz_layout(G, prog='dot')

        from matplotlib.colors import Normalize, LinearSegmentedColormap

        blue_orange = LinearSegmentedColormap.from_list(
            'blue_orange',
            ['tab:blue', 'lightgray', 'tab:orange']
        )

        inf = float('Inf')
        x_range = [inf, -inf]
        y_range = [inf, -inf]
        for id_node, pos in positions.items():
            x, y = pos
            x_range = [min(x, x_range[0]), max(x, x_range[1])]
            y_range = [min(y, y_range[0]), max(y, y_range[1])]

            player = nodes_extra[id_node]['player']
            text_player = 'p:{}'.format(player)
            text_depth = 'd:{}'.format(nodes_extra[id_node]['depth'])
            color_player = 'tab:blue' if player == -1 else 'tab:orange'

            frameon = False
            bboxprops = None
            if nodes_extra[id_node]['terminal']:
                winner = nodes_extra[id_node]['winner']
                frameon = True
                if winner is None:
                    edgecolor = 'tab:purple'
                else:
                    edgecolor = 'tab:blue' if winner == -1 else 'tab:orange'
                bboxprops = dict(
                    facecolor='none',
                    edgecolor=edgecolor
                )
                color_player = 'k'
                text_player = 'w:{}'.format(winner)
                if winner is None:
                    text_player = ''

            # needs to be transposed b/c image coordinates etc ...
            board = nodes_extra[id_node]['board'].T
            textbox_player = TextArea(text_player, textprops=dict(size=6, color=color_player))
            textbox_depth = TextArea(text_depth, textprops=dict(size=6))

            textbox_children = [textbox_player, textbox_depth]

            if highlight is not None:
                if id_node in highlight:
                    if nodes_extra[id_node]['terminal']:
                        frameon = True
                        if nodes_extra[id_node]['winner'] is None:
                            edgecolor = 'tab:purple'
                        else:
                            edgecolor = 'tab:blue' if winner == -1 else 'tab:orange'

                        bboxprops = dict(
                            facecolor='none',
                            edgecolor=edgecolor
                        )

                    if len(highlight[id_node]) > 0:
                        for key, value in highlight[id_node].items():
                            textbox_children.append(
                                TextArea('{}:{}'.format(key, value), textprops=dict(size=6))
                            )

            imagebox = OffsetImage(board, zoom=5, cmap=blue_orange, norm=Normalize(vmin=-1, vmax=1))
            packed = HPacker(
                align='center',
                children=[
                    imagebox,
                    VPacker(
                        align='center',
                        children=textbox_children,
                        sep=0.1, pad=0.1
                    )
                ],
                sep=0.1, pad=0.1
            )

            ab = AnnotationBbox(packed, pos, xycoords='data', frameon=frameon, bboxprops=bboxprops)
            tree_ax.add_artist(ab)

        def min_dist(a, b):
            if a == b:
                return [a - 1, b + 1]
            else:
                return [a - 0.9 * abs(a), b + 0.1 * abs(b)]

        x_range = min_dist(*x_range)
        y_range = min_dist(*y_range)
        tree_ax.set_xlim(x_range)
        tree_ax.set_ylim(y_range)

        orange_edges = []
        blue_edges = []

        for edge, extra in edges_extra.items():
            if extra['parent_player'] == -1:
                blue_edges.append(edge)
            else:
                orange_edges.append(edge)

        for color, edgelist in [('tab:orange', orange_edges), ('tab:blue', blue_edges)]:
            nx.draw_networkx_edges(
                G, positions,
                edgelist=edgelist,
                edge_color=color,
                arrowstyle='-|>',
                arrowsize=10,
                node_size=node_size,
                ax=tree_ax
            )
        edge_labels = {edge_id: edge['label'] for edge_id, edge in edges_extra.items()}
        nx.draw_networkx_edge_labels(G, positions, edge_labels, ax=tree_ax, font_size=6)

        tree_ax.axis('off')
        plt.tight_layout()
        plt.show()