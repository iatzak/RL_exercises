import gym
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        # through state history and check if state has occurred before
        for s in states:
            returns[s[0]-12, s[1]-1, int(s[2])] += G
            visits[s[0]-12, s[1]-1, int(s[2])] += 1
            V[s[0]-12, s[1]-1, int(s[2])] = returns[s[0]-12, s[1]-1, int(s[2])] / visits[s[0]-12, s[1]-1, int(s[2])]

    player_sums = np.array(range(12, 22))
    dealer_cards = np.array(range(1, 11))

    state_values = V[:, :, 0]
    fig = go.Figure(go.Surface(x = dealer_cards, y = player_sums, z=state_values))
    fig.update_layout(title=f"Monte Carlo policy evaluation for blackjack, after {maxiter} episodes",
                      scene=dict(
                        yaxis_title='Player Sum',
                        xaxis_title='Dealer Showing',
                        zaxis_title='State Value'
                        ),
                      font=dict(
                        family="Courier New, monospace",
                        size=20
                        )
                      )
    fig.show()

    state_values = V[:, :, 1]
    fig = go.Figure(go.Surface(x = dealer_cards,y = player_sums,z=state_values))
    fig.update_layout(title="Monte Carlo First Visit for Blackjack",
                      scene=dict(
                        yaxis_title='Player Sum',
                        xaxis_title='Dealer Showing',
                        zaxis_title='State Value'
                        ),
                      font=dict(
                        family="Courier New, monospace",
                        size=20
                        )
                      )
    fig.show()


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2))
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 100000000  # use whatever number of iterations you want
    for i in range(maxiter):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def main():
    # single_run_20()
    policy_evaluation()
    # monte_carlo_es()


if __name__ == "__main__":
    main()
