import numpy as np

class UCBBandit:
    def __init__(self, num_arms:int=10, mu:float=0, var:float=3, **params):
        """
        Args:
            num_arms (int, optional): Number of arms in the bandit. Defaults to 10.
            mu (float, optional): Mean of the normal distribution. Defaults to 0.
            var (float, optional): Variance of the normal distribution. Defaults to 3.
        """
        self.num_arms = num_arms
        # Trade-off parameter between the two terms in UCB
        self.c = params.get('param', 2)
        self.mu = mu
        self.var = var
        self.arm_values = [0] * num_arms
        self.arm_counts = [0] * num_arms
        self.averages = np.random.normal(self.mu, self.var, self.num_arms)
        self.total_rewards = []
      
    def pull_arm(self, i):
        # Estimate an upper bound on the true action values
        # Select the action with the largest (estimated) upper bound
        # We add c * sqrt(ln(t) / Nt(a)) to the estimate to ensure exploration
        # self.arm_values ensures exploitation
        action = np.argmax(self.arm_values + self.c*np.sqrt((np.log(i))/self.arm_counts))
        return action
        
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
            chosen_arm = self.pull_arm(i)
            # reward is pulled from a normal distribution using the arm's average as the mean and 1 as the variance
            reward = np.random.normal(self.averages[chosen_arm], 1)
            self.update(chosen_arm, reward)
            self.total_rewards.append(reward)
        
        return self.total_rewards
            

