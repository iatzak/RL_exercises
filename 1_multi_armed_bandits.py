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


def select_action(bandit, timesteps, eps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    for arm in possible_arms:
        rewards[arm] += bandit.play_arm(arm)
        n_plays[arm] += 1
        Q[arm] = rewards[arm] / n_plays[arm]

    while bandit.total_played < timesteps:
        if random.random() < eps:
            arm = random.randint(0, bandit.n_arms - 1)
        else:
            arm = np.argmax(Q)
        rewards[arm] += bandit.play_arm(arm)
        n_plays[arm] += 1
        Q[arm] = rewards[arm] / n_plays[arm]


def main():
    n_episodes = 10000
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        select_action(b, n_timesteps, eps=0.)  # greedy action selection
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        select_action(b, n_timesteps, eps=0.1)  # epsilon-greedy action selection
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    # plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
