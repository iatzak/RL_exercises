import gym
import numpy as np

custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
# env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# Init environment
# env = gym.make("FrozenLake-v0")

# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)
# Or:
env = gym.make("FrozenLake-v0", map_name="8x8")


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def print_policy(policy, env):
    """ This is a helper function to print a nice policy representation from the policy"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    pol = np.chararray(dims, unicode=True)
    pol[:] = ' '
    for s in range(len(policy)):
        idx = np.unravel_index(s, dims)
        pol[idx] = moves[policy[s]]
        if env.desc[idx] in [b'H', b'G']:
            pol[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in pol]))


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    policy = np.zeros(n_states, dtype=int)

    delta = theta + 1
    i = 0
    while delta > theta:
        delta = 0.
        for s in range(n_states):
            v = V_states[s]
            max_sum = 0.
            for a in range(n_actions):
                bellman_sum = 0
                for p, s_prime, r, _ in env.P[s][a]:
                    bellman_sum += p*(r + gamma*V_states[s_prime])
                if bellman_sum > max_sum:
                    max_sum = bellman_sum
                    policy[s] = a
            V_states[s] = max_sum
            delta = max(delta, np.abs(v - V_states[s]))
        i += 1
    print(f"Number of iterations: {i}")
    print(f"Optimal value function:\n{np.array2string(V_states, precision=3, floatmode='fixed')}\n")
    return policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    dims = env.desc.shape
    print()

    # run the value iteration
    policy = value_iteration()
    print("Computed policy: ")
    print(policy.reshape(dims))
    # if you computed a (working) policy, you can print it nicely with the following command:
    print_policy(policy, env)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break"""


if __name__ == "__main__":
    main()
