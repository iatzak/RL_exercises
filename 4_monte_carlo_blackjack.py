import gym
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def single_run_20(env: gym.Env):
    """Run the policy that sticks for player sum >= 20."""
    state = env.reset()  # state is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    G = 0.
    while not done:
        states.append(state)
        if state[0] >= 20:
            state, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            state, reward, done, _ = env.step(1)  # step=1 for hit
        G += reward  # gamma = 1
    return states, G


def single_run_ES(pi: np.ndarray, env: gym.Env):
    """Run policy that's being improved with random initial state-action pair."""
    done = False
    s0 = env.reset()  # Random initial state
    a0 = np.random.randint(0, 2)  # Random initial action
    states = [s0]  # Store states
    actions = [a0]  # Store actions
    state, reward, done, _ = env.step(a0)  # Initial step
    G = reward  # Initialize return G
    while not done:
        states.append(state)
        action = pi[(state[0]-12, state[1]-1, int(state[2]))]  # After initial action, follow pi
        actions.append(action)
        state, reward, done, _ = env.step(action)
        G += reward  # gamma = 1
    return states, actions, G


def plot_state_values(state_values: np.ndarray, max_iter: int):
    """Plot the state values obtained from Monte Carlo policy evaluation."""
    player_sums = np.array(range(12, 22))
    dealer_cards = np.array(range(1, 11))

    # Initialize subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['With usable ace', 'Without usable ace'],
                        horizontal_spacing=0.01,  vertical_spacing=0.1
                        )

    # Add data to subplots
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=state_values[:, :, 1], coloraxis='coloraxis'), row=1, col=1)
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=state_values[:, :, 0], coloraxis='coloraxis'), row=1, col=2)

    # Update axes titles
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

    # Update figure layout
    fig.update_layout(title=f"Monte Carlo policy evaluation for blackjack (stick if sum>=20), after {max_iter} episodes",
                      font=dict(family="Courier New, monospace", size=16),
                      title_font_size=30,
                      coloraxis=dict(colorscale='thermal')
                      )
    fig.update_annotations(font_size=26)  # for subplot titles

    # Display figure
    fig.show()


def plot_optimal_policy_and_value_function(pi: np.ndarray, V: np.ndarray, max_iter: int):
    """Plot the optimal policy and value function obtained from Monte Carlo ES."""
    player_sums = np.array(range(12, 22))
    dealer_cards = np.array(range(1, 11))

    policy_with_usable_ace = pi[:, :, 1]
    policy_without_usable_ace = pi[:, :, 0]
    V_with_usable_ace = V[:, :, 1]
    V_without_usable_ace = V[:, :, 0]

    # Initialize subplots
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{}, {'is_3d': True}],
                               [{}, {'is_3d': True}]],
                        column_titles=['$\pi^*$', '$v^*$'],
                        row_titles=['with\nusable\nace', 'without usable ace'],
                        column_widths=[0.5, 0.5],
                        horizontal_spacing=0.01,  vertical_spacing=0.1
                        )

    # Add data to subplots
    fig.add_trace(go.Heatmap(z=policy_with_usable_ace, colorscale=[[0, 'white'], [1, 'black']], showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=policy_without_usable_ace, colorscale=[[0, 'white'], [1, 'black']], showscale=False), row=2, col=1)
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=V_with_usable_ace, coloraxis='coloraxis'), row=1, col=2)
    fig.add_trace(go.Surface(x=dealer_cards, y=player_sums, z=V_without_usable_ace, coloraxis='coloraxis'), row=2, col=2)

    # Update policy axes with player sum and dealer card
    fig.update_xaxes(dict(
        tickmode='array',
        tickvals=list(range(10)),
        ticktext=['A', 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ), row=1, col=1)
    fig.update_xaxes(dict(
        tickmode='array',
        tickvals=list(range(10)),
        ticktext=['A', 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ), row=2, col=1)
    fig.update_yaxes(dict(
        tickmode='array',
        tickvals=list(range(10)),
        ticktext=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        ), row=1, col=1)
    fig.update_yaxes(dict(
        tickmode='array',
        tickvals=list(range(10)),
        ticktext=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        ), row=2, col=1)

    # Write text in policy plots to identify actions
    fig.add_annotation(x=4, y=2, text='HIT', showarrow=False, font=dict(color='white'),row=1, col=1)
    fig.add_annotation(x=4, y=8, text='STICK', showarrow=False, row=1, col=1)
    fig.add_annotation(x=7, y=2, text='HIT', showarrow=False, font=dict(color='white'),row=2, col=1)
    fig.add_annotation(x=4, y=8, text='STICK', showarrow=False, row=2, col=1)

    # Update value function axes titles
    fig.update_scenes(dict(
        yaxis_title='Player Sum',
        xaxis_title='Dealer Showing',
        zaxis_title='State Value'
        ), row=1, col=2)
    fig.update_scenes(dict(
        yaxis_title='Player Sum',
        xaxis_title='Dealer Showing',
        zaxis_title='State Value'
        ), row=2, col=2)

    # Update figure layout
    fig.update_layout(title=f"Optimal policy and state-value function found by Monte Carlo ES with {max_iter} iterations",
                      font=dict(family="Courier New, monospace", size=16),
                      title_font_size=30,
                      coloraxis=dict(colorscale='thermal')
                      )
    fig.update_annotations(font_size=26)  # for subplot titles

    # Display figure
    fig.show()


def policy_evaluation(env: gym.Env, max_iter: int):
    """Evaluate policy stick if player sum >= 20 with first-visit Monte Carlo"""
    # Dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    for i in range(max_iter):
        states, G = single_run_20(env)
        # In blackjack, the same state never repeats within a game
        # Therefore, in this case it's not needed to loop backwards
        # through state history and check if state has occurred before;
        # I'm doing it for the sake of completeness of the exercise
        for t, state in sorted(enumerate(states), reverse=True):
            state_index = state[0]-12, state[1]-1, int(state[2])
            if state not in states[:t]:
                returns[state_index] += G
                visits[state_index] += 1
                V[state_index] = returns[state_index] / visits[state_index]  # Update approximation of state value

    # Plot results
    plot_state_values(V, max_iter)


def monte_carlo_ES(env: gym.Env, max_iter: int):
    """ Implementation of Monte Carlo ES """
    # Dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    pi = np.zeros((10, 10, 2), dtype=int)
    Q = np.ones((10, 10, 2, 2)) * 100  # Optimistic initialization of state-action value
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    for i in range(max_iter):
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
            state_index = (state[0]-12, state[1]-1, int(state[2]))
            if (state, action) not in zip(states[:t], actions[:t]):
                returns[state_index][action] += G
                visits[state_index][action] += 1
                # Update approximation of action value function and policy
                Q[state_index][action] = returns[state_index][action] / visits[state_index][action]
                pi[state_index] = np.argmax(Q[state_index], axis=-1)

    # Get optimal value function from action value function
    V = np.max(Q, axis=-1)

    # Plot results
    plot_optimal_policy_and_value_function(pi, V, max_iter)


def main():
    # Initialite Blackjack environment
    env = gym.make('Blackjack-v0')

    max_iter = int(1e7)

    # policy_evaluation(env, max_iter)
    monte_carlo_ES(env, max_iter)


if __name__ == "__main__":
    main()
