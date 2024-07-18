import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and variance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def run_episode(bandit, timesteps, eps, annealing=False):
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    for arm in possible_arms:
        Q[arm], n_plays[arm] = update_estimate_of_action_value(bandit, Q[arm], n_plays[arm], arm)

    while bandit.total_played < timesteps:
        if annealing:
            eps *= 0.999

        if random.random() < eps:
            arm = random.randint(0, bandit.n_arms - 1)
        else:
            arm = get_argmax_random_tie_break(Q)

        Q[arm], n_plays[arm] = update_estimate_of_action_value(bandit, Q[arm], n_plays[arm], arm)


def get_argmax_random_tie_break(Q):
    """Argmax with random tie break."""
    indices_max_value = np.where(Q == Q.max())[0]
    return np.random.choice(indices_max_value)


def update_estimate_of_action_value(bandit, Q, n_plays, arm):
    """
    Calculate the updated estimate of the action value incrementally,
    using the general form
    NewEstimate = OldEstimate + StepSize[Target - OldEstimate],
    derived in section 2.4 of Sutton & Barto's book.
    """
    n_plays += 1
    step_size = 1/n_plays
    current_reward = bandit.play_arm(arm)
    Q += step_size * (current_reward - Q)
    return Q, n_plays


def main():
    n_episodes = 1000
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)
    rewards_egreedy_annealing = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        bandit = GaussianBandit()  # initializes a random bandit
        for strategy, rewards in zip([(0., False), (0.1, False), (0.1, True)],
                                     [rewards_greedy, rewards_egreedy, rewards_egreedy_annealing]):
            bandit.reset()
            run_episode(bandit, n_timesteps, eps=strategy[0], annealing=strategy[1])
            rewards += bandit.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    rewards_egreedy_annealing /= n_episodes

    for strategy_name, rewards in zip(['Greedy', 'Epsilon-Greedy', 'Epsilon-Greedy with Annealing'],
                                      [rewards_greedy, rewards_egreedy, rewards_egreedy_annealing]):
        total_reward = np.sum(rewards)
        print(f"Total reward of {strategy_name} strategy averaged over {n_episodes} episodes: {total_reward}")

    plt.plot(rewards_greedy, label="greedy")
    plt.plot(rewards_egreedy, label="e-greedy")
    plt.plot(rewards_egreedy_annealing, label="e-greedy w/ annealing")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Comparison of Action-Selecting Strategies for the k-armed Bandit Problem")
    # plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
