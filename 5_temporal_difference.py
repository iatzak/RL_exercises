import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import patches


def print_policy(Q, env):
    """Print a policy obtained from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = env.desc[idx]
    print("Policy:")
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in policy]))


def plot_V(Q, env, ax):
    """Plot the state values obtined from the Q function"""
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    ax.imshow(V, origin='upper',
              extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6,
              cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        ax.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                horizontalalignment='center',
                verticalalignment='center')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_Q(Q, env, ax):
    """Plot the action value function"""

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            ax.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                    horizontalalignment='center',
                    verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            ax.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    ax.set_xticks([])
    ax.set_yticks([])


def plot_episode_lengths(episode_lengths, ax):
    """Plot the episode lengths against the episode index"""
    ax.plot(episode_lengths)
    ax.set_title("Episode lengths")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")


def create_environment(env_name='FrozenLake-v0', is_slippery=True, map_name='4x4'):
    """Create and return the environment based on given parameters."""
    if env_name == 'FrozenLake-v0':
        return gym.make(env_name, is_slippery=is_slippery, map_name=map_name)
    else:
        return gym.make(env_name)


def bin_epsisode_lengths(episode_lengths, bin_size=100):
    """Group episodes lengths in bins"""
    reshaped_lengths = episode_lengths[:len(episode_lengths)
        // bin_size * bin_size].reshape(-1, bin_size)
    return reshaped_lengths.mean(axis=1)


def select_action_eps_greedy(env, Q, s, eps):
    """Get an action using the epsilon-greedy action selection"""
    if np.random.random() > eps:
        return np.argmax(Q[s])
    else:
        return np.random.randint(env.action_space.n)


def run_temporal_difference(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4), algorithm='sarsa'):
    """Calculate the action-value function with a temporal difference algorithm (SARSA or Q-learning)"""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_lengths = np.zeros(num_ep)

    for i in range(num_ep):
        s = env.reset()
        a = select_action_eps_greedy(env, Q, s, epsilon)
        done = False
        steps = 0
        while not done:
            s_prime, r, done, _ = env.step(a)
            a_prime = select_action_eps_greedy(env, Q, s_prime, epsilon)

            if algorithm == 'sarsa':
                target = r + gamma * Q[s_prime][a_prime]
            elif algorithm == 'qlearning':
                target = r + gamma * np.max(Q[s_prime])
            else:
                raise ValueError("Algorithm must be either 'sarsa' or 'qlearning'.")

            Q[s][a] += alpha*(target - Q[s][a])

            s, a = s_prime, a_prime
            steps += 1

        episode_lengths[i] = steps

    return Q, episode_lengths


env = create_environment(is_slippery=False)

print("Current Environment:")
env.render()

fig, ax = plt.subplots(nrows=2, ncols=3)

print("\nRunning SARSA")
Q, ep_lengths = run_temporal_difference(env, algorithm='sarsa')
binned_ep_lengths = bin_epsisode_lengths(ep_lengths)
plot_V(Q, env, ax[0][0])
plot_Q(Q, env, ax[0][1])
plot_episode_lengths(binned_ep_lengths, ax[0][2])
print_policy(Q, env)

print("\nRunning Q-learning")
Q, ep_lengths = run_temporal_difference(env, algorithm='qlearning')
binned_ep_lengths = bin_epsisode_lengths(ep_lengths)
plot_V(Q, env, ax[1][0])
plot_Q(Q, env, ax[1][1])
plot_episode_lengths(binned_ep_lengths, ax[1][2])
print_policy(Q, env)
plt.show()
