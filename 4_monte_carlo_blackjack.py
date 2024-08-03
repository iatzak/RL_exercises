import gym
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        # print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            # print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            # print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print("final observation:", obs)
    return states, ret


def single_run_ES(pi: np.ndarray, env: gym.Env):
    """Run policy that's being improved with random initial state-action pair."""
    done = False
    s0 = env.reset()  # Random initial state
    a0 = np.random.randint(0, 2)  # Random initial action
    states = [s0]  # Store states
    actions = [a0]  # Store actions
    obs, reward, done, _ = env.step(a0)  # Initial step
    G = reward  # Initialize return G
    while not done:
        states.append(obs)
        action = pi[(obs[0]-12, obs[1]-1, int(obs[2]))]  # After initial action, follow pi
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        G += reward  # gamma == 1
    return states, actions, G


def plot_state_values(state_values: np.ndarray, maxiter: int):
    """Plot the state values obtained from Monte Carlo policy evaluation."""
    player_sums = np.array(range(12, 22))
    dealer_cards = np.array(range(1, 11))

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['With usable ace', 'Without usable ace'],
                        horizontal_spacing=0.01,  vertical_spacing=0.1
                        )
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=state_values[:, :, 1], coloraxis='coloraxis'), row=1, col=1)
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=state_values[:, :, 0], coloraxis='coloraxis'), row=1, col=2)
    fig.update_scenes(dict(
                        yaxis_title='Player Sum',
                        xaxis_title='Dealer Showing',
                        zaxis_title='State Value'
                        ), row=1, col=1)
    fig.update_scenes(dict(
                        yaxis_title='Player Sum',
                        xaxis_title='Dealer Showing',
                        zaxis_title='State Value'
                        ), row=1, col=2)
    fig.update_layout(title=f"Monte Carlo policy evaluation for blackjack, after {maxiter} episodes",
                      font=dict(family="Courier New, monospace", size=16),
                      title_font_size=30,
                      coloraxis=dict(colorscale='thermal')
                      )
    fig.update_annotations(font_size=26)  # for subplot titles
    fig.show()


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 100000  # use whatever number of iterations you want
    for i in range(maxiter):
        states, G = single_run_20()
        # In blackjack, the same state never repeats within a game
        # Therefore, in this case it's not needed to loop backwards
        # through state history and check if state has occurred before;
        # I'm doing it for the sake of completeness of the exercise
        for t, s in sorted(enumerate(states), reverse=True):
            if s not in states[:t]:
                returns[s[0]-12, s[1]-1, int(s[2])] += G
                visits[s[0]-12, s[1]-1, int(s[2])] += 1
                V[s[0]-12, s[1]-1, int(s[2])] = returns[s[0]-12, s[1]-1, int(s[2])] / visits[s[0]-12, s[1]-1, int(s[2])]

    # Plot results
    plot_state_values(V, maxiter)


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2), dtype=int)
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 1000000  # use whatever number of iterations you want
    for i in range(maxiter):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])

        # Play a game following pi with random initial state-action pair
        states, actions, G = single_run_ES(pi, env)

        # In blackjack, the same state never repeats within a game
        # Therefore, in this case it's not needed to loop backwards
        # through state history and check if state has occurred before;
        # I'm doing it for the sake of completeness of the exercise
        for t, (state, action) in sorted(enumerate(zip(states, actions)), reverse=True):
            if (state, action) not in zip(states[:t], actions[:t]):
                returns[state[0]-12, state[1]-1, int(state[2]), action] += G
                visits[state[0]-12, state[1]-1, int(state[2]), action] += 1
                Q[state[0]-12, state[1]-1, int(state[2]), action] = \
                    returns[state[0]-12, state[1]-1, int(state[2]), action] \
                        / visits[state[0]-12, state[1]-1, int(state[2]), action]
                pi[state[0]-12, state[1]-1, int(state[2])] = \
                    np.argmax(Q[state[0]-12, state[1]-1, int(state[2])], axis=-1)

    V = np.max(Q, axis=-1)
    plot_state_values(V, maxiter)


def main():
    # single_run_20()
    policy_evaluation()
    # monte_carlo_es()


if __name__ == "__main__":
    main()
