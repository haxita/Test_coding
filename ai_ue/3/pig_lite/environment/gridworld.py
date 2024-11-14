import json
import numpy as np

from pig_lite.environment.base import Environment

DELTAS = [
    (-1, 0),
    (+1, 0),
    (0, -1),
    (0, +1)
]
NAMES = [
    'left',
    'right',
    'up',
    'down'
]

def sample(rng, elements):
    """ Samples an element of `elements` randomly. """
    csp = np.cumsum([elm[0] for elm in elements])
    idx = np.argmax(csp > rng.uniform(0, 1))
    return elements[idx]


class Gridworld(Environment):
    def __init__(self, seed, dones, rewards, starts):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.dones = dones
        self.rewards = rewards
        self.starts = starts

        self.__compute_P()

    def reset(self):
        """ Resets the environment of this gridworld to a randomly sampled start state. """
        _, self.state = sample(self.rng, self.starts)
        return self.state

    def step(self, action):
        """ Performs the action on the gridworld, where next state of environment is sampled based on self.P. """
        _, self.state, reward, done = sample(self.rng, self.P[self.state][action])
        return self.state, reward, done

    def get_n_actions(self):
        """ Returns the number of actions available in this gridworld. """
        return 4

    def get_n_states(self):
        """ Returns the number of states available in this gridworld. """
        return np.prod(self.dones.shape)

    def get_gamma(self):
        """ Returns discount factor gamma for this gridworld. """
        return 0.99

    def __compute_P(self):
        """ Computes and stores the transitions for this gridworld. """
        w, h = self.dones.shape

        def inbounds(i, j):
            """ Checks whether coordinates i and j are within the grid.  """
            return i >= 0 and j >= 0 and i < w and j < h

        self.P = dict()
        for i in range(0, w):
            for j in range(0, h):
                state = j * w + i
                self.P[state] = dict()

                if self.dones[i, j]:
                    for action in range(self.get_n_actions()):
                        # make it absorbing
                        self.P[state][action] = [(1, state, 0, True)]
                else:
                    for action, (dx, dy) in enumerate(DELTAS):
                        ortho_dir_probs = [
                            (0.8, dx, dy),
                            (0.1, dy, dx),
                            (0.1, -dy, -dx)
                        ]
                        transitions = []
                        for p, di, dj in ortho_dir_probs:
                            ni = i + di
                            nj = j + dj
                            if inbounds(ni, nj):
                                # we move
                                sprime = nj * w + ni
                                done = self.dones[ni, nj]
                                reward = self.rewards[ni, nj]
                                transitions.append((p, sprime, reward, done))
                            else:
                                # stay in the same state, b/c we bounced
                                sprime = state
                                done = self.dones[i, j]
                                reward = self.rewards[i, j]
                                transitions.append((p, sprime, reward, done))

                        self.P[state][action] = transitions

    def to_json(self):
        """ Converts and stores this gridworld to a JSON file. """
        return json.dumps(dict(
            type=self.__class__.__name__,
            seed=self.seed,
            dones=self.dones.tolist(),
            rewards=self.rewards.tolist(),
            starts=self.starts.tolist()
        ))

    @staticmethod
    def from_json(jsonstring):
        """ Loads given JSON file, and creates gridworld with information. """
        data = json.loads(jsonstring)
        return Gridworld(
            data['seed'],
            np.array(data['dones']),
            np.array(data['rewards']),
            np.array(data['starts'], dtype=np.int64),
        )

    @staticmethod
    def from_dict(data):
        """ Creates gridworld with information in given data-dictionary. """
        return Gridworld(
            data['seed'],
            np.array(data['dones']),
            np.array(data['rewards']),
            np.array(data['starts'], dtype=np.int64),
        )

    @staticmethod
    def get_random_instance(rng, size):
        """ Given random generator and problem size, generates Gridworld instance. """
        dones, rewards, starts = Gridworld.__generate(rng, size)
        return Gridworld(rng.randint(0, 2 ** 31), dones, rewards, starts)

    @staticmethod
    def __generate(rng, size):
        """ Helper function that retrieves dones, rewards, starts for Gridworld instance generation. """
        dones = np.full((size, size), False, dtype=bool)
        rewards = np.zeros((size, size), dtype=np.int8) - 1

        coordinates = []
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                coordinates.append((i, j))
        indices = np.arange(len(coordinates))

        chosen = rng.choice(indices, max(1, len(indices) // 10), replace=False)

        for c in chosen:
            x, y = coordinates[c]
            dones[x, y] = True
            rewards[x, y] = -100

        starts = np.array([[1, 0]])
        dones[-1, -1] = True
        rewards[-1, -1] = 100

        return dones, rewards, starts

    @staticmethod
    def get_minimum_problem_size():
        return 3

    def visualize(self, outcome, coords=None, grid=None):
        """ Visualisation function for gridworld; plots environment, policy, Q. """
        policy = None
        Q = None
        V = None
        if outcome is not None:
            if outcome.policy is not None:
                policy = outcome.policy

            if outcome.V is not None:
                V = outcome.V

            if outcome.Q is not None:
                Q = outcome.Q

        self._plot_environment_and_policy(policy, V, Q, show_coordinates=coords, show_grid=grid)

    def _plot_environment_and_policy(self, policy=None,V=None, Q=None, show_coordinates=False,
                                     show_grid=False, plot_filename=None, debug_info=False):
        """ Function that plots environment and policy. """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        dones_ax = axes[0, 0]
        rewards_ax = axes[0, 1]
        V_ax = axes[1, 0]
        Q_ax = axes[1, 1]

        dones_ax.set_title('Terminal States and Policy')
        dones_ax.imshow(self.dones.T, cmap='gray_r', vmin=0, vmax=4)

        rewards_ax.set_title('Immediate Rewards')
        rewards_ax.imshow(self.rewards.T, cmap='RdBu_r', vmin=-25, vmax=25)

        if len(policy) > 0:
            self._plot_policy(dones_ax, policy)

        w, h = self.dones.shape
        V_array = V.reshape(self.dones.shape).T
        V_ax.set_title('State Value Function $V(s)$')
        r = max(1e-13, np.max(np.abs(V_array)))
        V_ax.imshow(V_array.T, cmap='RdBu_r', vmin=-r, vmax=r)

        if debug_info:
            for s in range(len(V)):
                sy, sx = divmod(s, w)
                V_ax.text(sx, sy, f'{sx},{sy}:{s}',
                          color='w', fontdict=dict(size=6),
                          horizontalalignment='center', verticalalignment='center')

        Q_ax.set_title('State Action Value Function $Q(s, a)$')
        poly_patches_q_values = self._draw_Q(Q_ax, Q, debug_info)

        def format_coord(x, y):
            for poly_patch, q_value in poly_patches_q_values:
                if poly_patch.contains_point(Q_ax.transData.transform((x, y))):
                    return f'x:{x:4.2f} y:{y:4.2f} {q_value}'
            return f'x:{x:4.2f} y:{y:4.2f}'

        Q_ax.format_coord = format_coord

        for ax in [dones_ax, rewards_ax, V_ax, Q_ax]:
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
            s = self.dones.shape[0]
            ax.set_xticks(np.arange(0, s, 1))
            ax.set_yticks(np.arange(0, s, 1))

            # Minor ticks
            ax.set_xticks(np.arange(-.5, s, 1), minor=True)
            ax.set_yticks(np.arange(-.5, s, 1), minor=True)

        if show_grid:
            for color, ax in zip(['m', 'w', 'w'], [dones_ax, rewards_ax, V_ax]):
                # Gridlines based on minor ticks
                ax.grid(which='minor', color=color, linestyle='-', linewidth=1)

        plt.tight_layout()
        if plot_filename is not None:
            plt.savefig(plot_filename)
            plt.close(fig)
        else:
            plt.show()

    def _plot_policy(self, ax, policy):
        """ Function that plots policy. """
        w, h = self.dones.shape
        xs = np.arange(w)
        ys = np.arange(h)
        xx, yy = np.meshgrid(xs, ys)

        # we need a quiver for each of the four action
        quivers = list()
        for a in range(self.get_n_actions()):
            quivers.append(list())

        # we parse the textual description of the lake
        for s in range(self.get_n_states()):
            y, x = divmod(s, w)
            if self.dones[x, y]:
                for a in range(self.get_n_actions()):
                    quivers[a].append((0., 0.))
            else:
                for a in range(self.get_n_actions()):
                    wdx, wdy = DELTAS[a]
                    corrected = np.array([wdx, -wdy])
                    quivers[a].append(corrected * policy[s, a])

        # plot each quiver
        for quiver in quivers:
            q = np.array(quiver)
            ax.quiver(xx, yy, q[:, 0], q[:, 1], units='xy', scale=1.5)

    def _draw_Q(self, ax, Q, debug_info):
        """ Function that draws Q. """
        pattern = np.zeros(self.dones.shape)
        ax.imshow(pattern, cmap='gray_r')
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle, Polygon
        w, h = self.dones.shape

        r = max(1e-13, np.max(np.abs(Q)))
        norm = Normalize(vmin=-r, vmax=r)
        cmap = plt.get_cmap('RdBu_r')
        sm = ScalarMappable(norm, cmap)

        hover_polygons = []
        for state in range(len(Q)):
            qs = Q[state]
            # print('qs', qs)
            y, x = divmod(state, w)
            if self.dones[x, y]:
                continue
            y += 0.5
            x += 0.5

            dx = 1
            dy = 1

            ulx = (x - 1) * dx
            uly = (y - 1) * dy

            rect = Rectangle(
                xy=(ulx, uly),
                width=dx,
                height=dy,
                edgecolor='k',
                facecolor='none'
            )
            ax.add_artist(rect)

            mx = (x - 1) * dx + dx / 2.
            my = (y - 1) * dy + dy / 2.

            ul = ulx, uly
            ur = ulx + dx, uly
            ll = ulx, uly + dy
            lr = ulx + dx, uly + dy
            m = mx, my

            up = [ul, m, ur]
            left = [ul, m, ll]
            right = [ur, m, lr]
            down = [ll, m, lr]
            action_polys = [left, right, up, down]
            for a, poly in enumerate(action_polys):
                poly_patch = Polygon(
                    poly,
                    edgecolor='k',
                    linewidth=0.1,
                    facecolor=sm.to_rgba(qs[a])
                )
                if debug_info:
                    mmx = np.mean([x for x, y in poly])
                    mmy = np.mean([y for x, y in poly])
                    sss = '\n'.join(map(str, self.P[state][a]))
                    ax.text(mmx, mmy, f'{NAMES[a][0]}:{sss}',
                            fontdict=dict(size=5), horizontalalignment='center',
                            verticalalignment='center')

                hover_polygons.append((poly_patch, f'{NAMES[a]}:{qs[a]:4.2f}'))
                ax.add_artist(poly_patch)
        return hover_polygons
