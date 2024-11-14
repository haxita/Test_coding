from pig_lite.problem.base import Problem, Node
from pig_lite.instance_generation import enc
import json
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS

class BaseLevel():
    def __init__(self, rng, size) -> None:
        self.rng = rng
        self.size = size
        self.field = None
        self.costs = None
        self.start = None
        self.end = None

        self.initialize_level()

    def initialize_level(self):
        raise NotImplementedError()

    def get_field(self):
        return self.field
    
    def get_costs(self):
        return self.costs
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    

class MazeLevel(BaseLevel):
    # this method generates a random maze according to prim's randomized
    # algorithm
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_Prim.27s_algorithm

    def __init__(self, rng, size):
        super().__init__(rng, size)


    def initialize_level(self):

        self.field = np.full((self.size, self.size), enc.WALL, dtype=np.int8)
        self.costs = self.rng.randint(1, 5, self.field.shape, dtype=np.int8)

        self.start = (0, 0)

        self.deltas = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]
        self.random_walk()
        end = np.where(self.field == enc.SPACE)
        self.end = (int(end[0][-1]), int(end[1][-1]))

        self.replace_walls_with_high_cost_tiles()

    def replace_walls_with_high_cost_tiles(self):
        # select only coordinates of walls
        walls = np.where(self.field == enc.WALL)

        n_walls = len(walls[0])

        # replace about a tenth of the walls...
        to_replace = self.rng.randint(0, n_walls, n_walls // 9)

        # ... with space, but very *costly* space (it's trap!)
        for ri in to_replace:
            x, y = walls[0][ri], walls[1][ri]
            self.field[x, y] = enc.SPACE
            self.costs[x, y] = 9

    def random_walk(self):
        frontier = list()

        sx, sy = self.start
        self.field[sx, sy] = enc.SPACE
        frontier.extend(self.get_walls(self.start))

        while len(frontier) > 0:
            current, opposing = frontier[self.rng.randint(len(frontier))]

            cx, cy = current
            ox, oy = opposing
            if self.field[ox, oy] == enc.WALL:
                self.field[cx, cy] = enc.SPACE
                self.field[ox, oy] = enc.SPACE
                frontier.extend(self.get_walls(opposing))
            else:
                frontier.remove((current, opposing))

    def in_bounds(self, position):
        x, y = position
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    def get_walls(self, position):
        walls = []
        px, py = position
        for dx, dy in self.deltas:
            cx = px + dx
            cy = py + dy
            current = (cx, cy)

            ox = px + 2 * dx
            oy = py + 2 * dy
            opposing = (ox, oy)

            if (self.in_bounds(current) and self.field[cx, cy] == enc.WALL and self.in_bounds(opposing)):
                walls.append((current, opposing))
        return walls
    

# this is code taken from
# https://github.com/dandrino/terrain-erosion-3-ways/blob/master/util.py
# Copyright (c) 2018 Daniel Andrino
# (project is MIT licensed)
def fbm(shape, p, lower=-np.inf, upper=np.inf):
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    envelope = (np.power(freq_radial, p, where=freq_radial != 0) *
                (freq_radial > lower) * (freq_radial < upper))
    envelope[0][0] = 0.0
    phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
    return np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope))


class TerrainLevel(BaseLevel):
    def __init__(self, rng, size):
        super().__init__(rng, size)

    def initialize_level(self):

        self.field = np.full((self.size, self.size), enc.SPACE, dtype=np.int8)

        self.costs = fbm(self.field.shape, -2)
        self.costs -= self.costs.min()
        self.costs /= self.costs.max()
        self.costs *= 9
        self.costs += 1
        self.costs = self.costs.astype(int)

        self.start = (0, 0)
        self.end = (self.size - 1, self.size - 1)

        x = 0
        y = self.size - 1
        for i in range(0, self.size):
            self.field[x, y] = enc.WALL
            x += 1
            y -= 1

        self.replace_one_or_more_walls()

    def replace_one_or_more_walls(self):
        # select only coordinates of walls
        walls = np.where(self.field == enc.WALL)
        n_walls = len(walls[0])
        n_replace = self.rng.randint(1, max(2, n_walls // 5))
        to_replace = self.rng.randint(0, n_walls, n_replace)

        for ri in to_replace:
            x, y = walls[0][ri], walls[1][ri]
            self.field[x, y] = enc.SPACE


class RoomLevel(BaseLevel):
    def __init__(self, rng, size):
        super().__init__(rng, size)
        
    def initialize_level(self):  
        self.field = np.full((self.size, self.size), enc.SPACE, dtype=np.int8)
        self.costs = np.ones_like(self.field, dtype=np.float32)

        k = 1
        self.subdivide(self.field.view(), self.costs.view(), k, 0, 0)

        # such a *crutch*!
        # this 'repairs' dead ends. horrible stuff.
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                s = 0
                s += self.field[x - 1, y]
                s += self.field[x + 1, y]
                s += self.field[x, y - 1]
                s += self.field[x, y + 1]
                if self.field[x, y] == enc.SPACE and s >= 3:
                    self.field[x - 1, y] = enc.SPACE
                    self.field[x + 1, y] = enc.SPACE
                    self.field[x, y - 1] = enc.SPACE
                    self.field[x, y + 1] = enc.SPACE

        spaces = np.where(self.field == enc.SPACE)
        n_spaces = len(spaces[0])

        n_danger = self.rng.randint(3, 7)
        dangers = self.rng.choice(range(n_spaces), n_danger, replace=False)
        for di in dangers:
            rx, ry = np.unravel_index(di, (self.size, self.size))
            const = max(1., self.rng.randint(self.size // 5, self.size // 2))
            for x in range(self.size):
                for y in range(self.size):
                    distance = np.sqrt((rx - x) ** 2 + (ry - y) ** 2)
                    self.costs[x, y] = self.costs[x, y] + (1. / (const + distance))

        self.costs = self.costs - self.costs.min()
        self.costs = self.costs / self.costs.max()
        self.costs = self.costs * 9
        self.costs = self.costs + 1
        self.costs = self.costs.astype(int)

        start_choice = 0
        end_choice = -1

        self.start = (int(spaces[0][start_choice]), int(spaces[1][start_choice]))
        self.end = (int(spaces[0][end_choice]), int(spaces[1][end_choice]))

        if self.start == self.end:
            raise RuntimeError('should never happen')

    def subdivide(self, current, costs, k, d, previous_door):
        w, h = current.shape
        random_stop = self.rng.randint(0, 10) == 0 and d > 2
        if w <= 2 * k + 1 or h <= 2 * k + 1 or random_stop:
            return

        split = previous_door
        while split == previous_door:
            split = self.rng.randint(k, w - k)
        current[split, :] = enc.WALL
        door = self.rng.randint(k, h - k)
        current[split, door] = enc.SPACE

        self.subdivide(
            current[:split, :].T,
            costs[:split, :].T,
            k,
            d + 1,
            door
        )
        self.subdivide(
            current[split + 1:, :].T,
            costs[split + 1:, :].T,
            k,
            d + 1,
            door
        )


class Simple2DProblem(Problem):
    """
    the states are the positions on the board that the agent can walk on
    """

    ACTIONS_DELTA = OrderedDict([
        ('R', (+1, 0)),
        ('U', (0, -1)),
        ('D', (0, +1)),
        ('L', (-1, 0)),
    ])

    def __init__(self, board, costs, start, end):
        self.board = board
        self.costs = costs
        self.start_state = start
        self.end_state = end
        self.n_expands = 0

    def get_start_node(self):
        return Node(None, self.start_state, None, 0, 0)

    def get_end_node(self):
        return Node(None, self.end_state, None, 0, 0)

    def is_end(self, node):
        return node.state == self.end_state

    def action_cost(self, state, action):
        # for the MazeProblem, the cost of any action
        # is stored at the coordinates of the successor state,
        # and represents the cost of 'stepping onto' this
        # position on the board
        sx, sy = self.__delta_state(state, action)
        return self.costs[sx, sy]

    def successor(self, node, action):
        # determine the next state
        successor_state = self.__delta_state(node.state, action)
        if successor_state is None:
            return None

        # determine what it would cost to take this action in this state
        cost = self.action_cost(node.state, action)

        # add the next state to the list of successor nodes
        return Node(
            node,
            successor_state,
            action,
            node.cost + cost,
            node.depth + 1
        )

    def get_number_of_expanded_nodes(self):
        return self.n_expands
    
    def reset(self):
        self.n_expands = 0

    def successors(self, node):
        self.n_expands += 1
        successor_nodes = []
        for action in self.ACTIONS_DELTA.keys():
            succ = self.successor(node, action)
            if succ is not None and succ != node:
                successor_nodes.append(succ)
        return successor_nodes

    def to_json(self):
        return json.dumps(dict(
            type=self.__class__.__name__,
            board=self.board.tolist(),
            costs=self.costs.tolist(),
            start_state=self.start_state,
            end_state=self.end_state
        ))
    
    @staticmethod
    def draw_nodes(fig, ax, name, node_collection, color, marker):
        states = np.array([node.state for node in node_collection])
        if len(states) > 0:
            ax.scatter(states[:, 0], states[:, 1], color=color, label=name, marker=marker)

    @staticmethod
    def plot_nodes(fig, ax, nodes):
        if len(nodes) > 0:
            if len(nodes[0]) == 3:
                for (name, marker, node_collection), color in zip(nodes, TABLEAU_COLORS):
                    if len(node_collection) > 0:
                        Simple2DProblem.draw_nodes(fig, ax, name, node_collection, color, marker)
            else:
                for name, marker, node_collection, color in nodes:
                    if len(node_collection) > 0:
                        Simple2DProblem.draw_nodes(fig, ax, name, node_collection, color, marker)

            ax.legend(
                bbox_to_anchor=(0.5, -0.03),
                loc='upper center',
            )
    
    def plot_sequences(self, fig, ax, sequences):
        start_node = self.get_start_node()
        for (name, action_sequence), color in zip(sequences, XKCD_COLORS):
            self.draw_path(fig, ax, name, start_node, action_sequence, color)

        ax.legend(
            bbox_to_anchor=(0.5, -0.03),
            loc='upper center',
        )


    def draw_path(self, fig, ax, name, start_node, action_sequence, color):
        current = start_node
        xs = [current.state[0]]
        ys = [current.state[1]]
        us = [0]
        vs = [0]

        length = len(action_sequence)
        cost = 0
        costs = [0] * length
        for i, action in enumerate(action_sequence):
            costs[i] = current.cost
            xs.append(current.state[0])
            ys.append(current.state[1])
            current = self.successor(current, action)
            dx, dy = self.ACTIONS_DELTA[action]
            us.append(dx)
            vs.append(-dy)
            cost = current.cost

        quiv = ax.quiver(
            xs, ys, us, vs,
            color=color,
            label='{} l:{} c:{}'.format(name, length, cost),
            scale_units='xy',
            units='xy',
            scale=1,
            headwidth=1,
            headlength=1,
            linewidth=1,
            picker=5
        )
        return quiv

    def plot_field_and_costs_aux(self, fig, show_coordinates, show_grid,
                             field_ax=None, costs_ax=None):

        if field_ax is None:
            ax = field_ax = plt.subplot(121)
        else:
            ax = field_ax

        ax.set_title('The field')
        im = ax.imshow(self.board.T, cmap='gray_r')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([0, 1])

        if costs_ax is None:
            ax = costs_ax = plt.subplot(122, sharex=ax, sharey=ax)
        else:
            ax = costs_ax

        ax.set_title('The costs (for stepping on a tile)')
        im = ax.imshow(self.costs.T, cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        ticks = np.arange(self.costs.min(), self.costs.max() + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

        for ax in [field_ax, costs_ax]:
            ax.tick_params(
                top=show_coordinates,
                left=show_coordinates,
                labelleft=show_coordinates,
                labeltop=show_coordinates,
                right=False,
                bottom=False,
                labelbottom=False
            )

            # Major ticks
            s = self.board.shape[0]
            ax.set_xticks(np.arange(0, s, 1))
            ax.set_yticks(np.arange(0, s, 1))

            # Minor ticks
            ax.set_xticks(np.arange(-.5, s, 1), minor=True)
            ax.set_yticks(np.arange(-.5, s, 1), minor=True)

        if show_grid:
            for color, ax in zip(['m', 'w'], [field_ax, costs_ax]):
                # Gridlines based on minor ticks
                ax.grid(which='minor', color=color, linestyle='-', linewidth=1)

        return field_ax, costs_ax

    def visualize(self, sequences=None, show_coordinates=False, show_grid=False, plot_filename=None):
        
        nodes = [
            ('start', 'o', [self.get_start_node()]),
            ('end', 'o', [self.get_end_node()])
        ]

        fig = plt.figure(figsize=(10, 7))
        field_ax, costs_ax = self.plot_field_and_costs_aux(fig, show_coordinates, show_grid)
        if sequences is not None and len(sequences) > 0:
            self.plot_sequences(fig, field_ax, sequences)
            self.plot_sequences(fig, costs_ax, sequences)

        if nodes is not None and len(nodes) > 0:
            Simple2DProblem.plot_nodes(fig, field_ax, nodes)

        plt.tight_layout()
        if plot_filename is not None:
            plt.savefig(plot_filename)
            plt.close(fig)
        else:
            plt.show()


    @staticmethod
    def from_json(jsonstring):
        data = json.loads(jsonstring)
        return Simple2DProblem(
            np.array(data['board']),
            np.array(data['costs']),
            tuple(data['start_state']),
            tuple(data['end_state'])
        )

    @staticmethod
    def from_dict(data):
        return Simple2DProblem(
            np.array(data['board']),
            np.array(data['costs']),
            tuple(data['start_state']),
            tuple(data['end_state'])
        )

    def __delta_state(self, state, action):
        # the old state's coordinates
        x, y = state

        # the deltas for each coordinates
        dx, dy = self.ACTIONS_DELTA[action]

        # compute the coordinates of the next state
        sx = x + dx
        sy = y + dy

        if self.__on_board(sx, sy) and self.__walkable(sx, sy):
            # (sx, sy) is a *valid* state if it is on the board
            # and there is no wall where we want to go
            return sx, sy
        else:
            # EIEIEIEIEI. up until assignment 1, this returned None :/
            # this had no consequences on the correctness of the algorithms,
            # but the explanations, and the self-edges were wrong
            return x, y

    def __on_board(self, x, y):
        size = len(self.board)  # all boards are quadratic
        return x >= 0 and x < size and y >= 0 and y < size

    def __walkable(self, x, y):
        return self.board[x, y] != enc.WALL
