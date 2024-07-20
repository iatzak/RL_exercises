import gym
import numpy as np
from itertools import product

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# Uncomment the following line to try the default map (4x4):
# env = gym.make("FrozenLake-v0")

# Uncomment the following lines for larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Initialize useful variables
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states)  # rewards are zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


def trans_matrix_for_policy(policy):
    """Return the transition probability matrix P for a policy"""
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


def terminals():
    """Return a list with the indices of the terminal states"""
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    """Calculate the value function vector by solving (I-gamma*P)v=r"""
    P = trans_matrix_for_policy(policy)
    I = np.eye(len(P))
    A = I - gamma*P
    v = np.linalg.solve(A, r)
    return v


def bruteforce_policies():
    """Determine the optimal policies by comparing the value functions of all possible policies"""
    indices_terminal_states = terminals()
    indices_non_terminal_states = np.setdiff1d(range(n_states), indices_terminal_states)

    # Generate every policy, i.e. all n_states-sized arrays with every possible combination of range(n_actions)
    all_policies = list(product(range(n_actions), repeat=n_states))
    # Calculate the value function of every policy
    all_values = [value_policy(policy) for policy in all_policies]

    # Initialize variables to store results
    optimal_policies_indices = []
    unique_policies_non_terminal = {}
    max_value = -np.inf

    # An optimal policy maximizes the value of each state
    # Thus, it suffices to check for the sum of each policy value
    # i.e., there's no need to check for the value of each state
    for index, values in enumerate(all_values):
        sum_values = np.sum(values)
        policy = all_policies[index]
        # Store non-terminal entries of policy, since policies that differ
        # only by terminal states are redundant
        # Converted to tuple to be hashable for dict
        policy_non_terminal = tuple([policy[i] for i in indices_non_terminal_states])

        # If a new optimal policy is found, list is reinitialized with its corresponding index only
        # Dictionary is also reinitialized with non-terminal actions and the index
        if sum_values > max_value:
            max_value = sum_values
            optimal_policies_indices = [index]
            unique_policies_non_terminal = {policy_non_terminal: index}

        # If a optimal policy with equal value is found, compare non-terminal
        # actions: if they are the same, then the policy is redundant,
        # if not, then the optimal policy is new and is thus stored
        elif sum_values == max_value:
            if policy_non_terminal not in unique_policies_non_terminal.keys():
                optimal_policies_indices.append(index)
                unique_policies_non_terminal[policy_non_terminal] = index

    # Get optimal policies (now including terminal states for full visualization)
    optimal_policies = [all_policies[i] for i in optimal_policies_indices]
    # Get optimal value function. Use any optimal policy, since their values are equal
    optimal_value = value_policy(optimal_policies[0])

    print(f"Optimal value function:\n{np.array2string(optimal_value, precision=3, floatmode='fixed')}\n")
    print(f"Number of optimal policies:\n{len(optimal_policies)}\n")
    print(f"Optimal policies:\n{np.array(optimal_policies)}\n")
    return optimal_policies


def main():
    print("Current environment:")
    env.render()
    print("")

    # Test simple policies
    policy_left = np.zeros(n_states, dtype=np.int64)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int64) * 2  # 2 for all states

    # Print value functions for simple policies
    print(f"Value function for policy_left (always going left):\n{np.array2string(value_policy(policy_left), precision=3, floatmode='fixed')}\n")
    print(f"Value function for policy_right (always going right):\n{np.array2string(value_policy(policy_right), precision=3, floatmode='fixed')}\n")

    optimalpolicies = bruteforce_policies()

    # This code can be used to "rollout" a policy in the environment:
    """
    print("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break"""


if __name__ == "__main__":
    main()
