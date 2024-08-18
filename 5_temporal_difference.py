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


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4)):
    """Calculate the action-value function with the SARSA algorithm"""
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
            Q[s][a] += alpha*(r + gamma*Q[s_prime][a_prime] - Q[s][a])
            s = s_prime
            a = a_prime
            steps += 1
        episode_lengths[i] = steps

    return Q, episode_lengths


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4)):
    """Calculate the action-value function with the Q-learning algorithm"""
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
            Q[s][a] += alpha*(r + gamma*np.max(Q[s_prime]) - Q[s][a])
            s = s_prime
            a = a_prime
            steps += 1
        episode_lengths[i] = steps

    return Q, episode_lengths


# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake-v0', is_slippery=False)
# env = gym.make('FrozenLake-v0', map_name="8x8", is_slippery=False)

print("Current Environment:")
env.render()

fig, ax = plt.subplots(nrows=2, ncols=3)

print("\nRunning SARSA")
Q, ep_lengths = sarsa(env)
binned_ep_lengths = bin_epsisode_lengths(ep_lengths)
plot_V(Q, env, ax[0][0])
plot_Q(Q, env, ax[0][1])
plot_episode_lengths(binned_ep_lengths, ax[0][2])
print_policy(Q, env)

print("\nRunning Q-learning")
Q, ep_lengths = qlearning(env)
binned_ep_lengths = bin_epsisode_lengths(ep_lengths)
plot_V(Q, env, ax[1][0])
plot_Q(Q, env, ax[1][1])
plot_episode_lengths(binned_ep_lengths, ax[1][2])
print_policy(Q, env)
plt.show()
