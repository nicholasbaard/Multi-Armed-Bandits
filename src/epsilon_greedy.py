import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, num_arms:int=10, mu:float=0, var:float=3, **params):
        """
        This class implements the epsilon greedy bandit algorithm. The algorithm uses the parameter epsilon to determine
        the probability of choosing a random arm.

        Args:
            num_arms (int, optional): Number of arms in the bandit. Defaults to 10.
            mu (float, optional): Mean of the normal distribution. Defaults to 0.
            var (float, optional): Variance of the normal distribution. Defaults to 3.
        """
        self.num_arms = num_arms
        # Probability of choosing a random arm (exploration)
        self.epsilon = params.get('param', 0.1)
        self.mu = mu
        self.var = var
        self.arm_values = [0] * num_arms
        self.arm_counts = [0] * num_arms
        self.averages = np.random.normal(self.mu, self.var, self.num_arms)
        self.total_rewards = []
      
    def pull_arm(self):
        if np.random.random() < self.epsilon:
            # Explore - Take a random choice of all the arms
            return np.random.randint(0, self.num_arms - 1)
        else:
            # Exploit - Take a random choice of all the arms that have the max value (Q)
            max_value = max(self.arm_values)
            max_indices = [i for i, v in enumerate(self.arm_values) if v == max_value]
            return np.random.choice(max_indices)
        
    def update(self, chosen_arm:int, reward:float):
        # Update Q of the chosen arm 
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]
        Q = self.arm_values[chosen_arm]
        # Note, we use (1/n) as the step size and not an alpha
        # NewEstimate <- OldEstimate + StepSize[Target - OldEstimate]
        Q_next = Q + (1/n)*(reward - Q)
        self.arm_values[chosen_arm] = Q_next

    def traverse(self, iterations:int):
        for i in range(iterations):
            chosen_arm = self.pull_arm()
            # reward is pulled from a normal distribution using the arm's average as the mean and 1 as the variance
            reward = np.random.normal(self.averages[chosen_arm], 1)
            self.update(chosen_arm, reward)
            self.total_rewards.append(reward)
        
        return self.total_rewards

            

