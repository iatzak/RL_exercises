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
    n_episodes = 500
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)
    rewards_egreedy_annealing = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        run_episode(b, n_timesteps, eps=0.)  # greedy action selection
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        run_episode(b, n_timesteps, eps=0.1)  # epsilon-greedy action selection
        rewards_egreedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        run_episode(b, n_timesteps, eps=0.1, annealing=True)  # epsilon-greedy with annealing
        rewards_egreedy_annealing += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    rewards_egreedy_annealing /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.plot(rewards_egreedy_annealing, label="e-greedy w/ annealing")
    print("Total reward of epsilon greedy strategy with annealing averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy_annealing)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Comparison of Action-Selecting Strategies for the k-armed Bandit Problem")
    # plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
