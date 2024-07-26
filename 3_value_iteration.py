import gym
import numpy as np

# Initial environment
env = gym.make("FrozenLake-v0")

# Deterministic environment
# env = gym.make("FrozenLake-v0", is_slippery=False)

# Larger environment
# env = gym.make("FrozenLake-v0", map_name="8x8")


# Useful variables
n_states = env.observation_space.n
n_actions = env.action_space.n


def print_policy(policy, env):
    """Print a representation of the policy"""
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
    """
    Calculate an approximation of an optimal policy using
    the value iteration algorithm as given in section 4.4 of
    Sutton and Barto
    """
    V_states = np.zeros(n_states)  # initialize all state values as 0
    theta = 1e-8  # threshold for accurary of estimation
    gamma = 0.8  # discount factor
    policy = np.zeros(n_states, dtype=int)  # placeholder

    i = 0  # counter for number of iterations
    while True:
        delta = 0.
        for s in range(n_states):
            v = V_states[s]  # previous state value
            max_sum = 0.
            for a in range(n_actions):  # get maximum value from all actions
                bellman_sum = 0
                for p, s_prime, r, _ in env.P[s][a]:
                    bellman_sum += p*(r + gamma*V_states[s_prime])  # Bellman update rule
                if bellman_sum > max_sum:
                    max_sum = bellman_sum
                    policy[s] = a
            V_states[s] = max_sum
            delta = max(delta, np.abs(v - V_states[s]))  # check if value function converged
        i += 1
        if delta < theta:
            break
    print(f"Number of iterations: {i}\n")
    print(f"Optimal value function:\n{np.array2string(V_states, precision=3, floatmode='fixed')}\n")
    return policy


def main():
    # Print the environment
    print("current environment: ")
    env.render()
    print()

    # Run the value iteration
    policy = value_iteration()
    print("Computed policy: ")
    # dims = env.desc.shape
    # print(policy.reshape(dims))
    print_policy(policy, env)

    # The code below can be used to "rollout" a policy in the environment
    """
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break
    """


if __name__ == "__main__":
    main()
