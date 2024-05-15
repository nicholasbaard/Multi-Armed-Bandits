import matplotlib.pyplot as plt

def run_bandits(num_iterations:int, 
                num_runs:int, 
                num_arms:int, 
                mu:float, 
                var:float, 
                bandit_class:str, 
                params:list
                ):
        """
        This function runs the bandit algorithm for each specified parameter.

        Returns:
            dict[list[float]]: A dictionary of the rewards for each parameter averaged over num_runs.
        """
        bandit_dict = {}
        for param in params:
            rewards = [0] * num_iterations
            for _ in range(num_runs):
                bandit = bandit_class(num_arms=num_arms, mu=mu, var=var, param=param)
                total_reward = bandit.traverse(num_iterations)
                rewards= [x + y for x, y in zip(rewards, total_reward)]

            bandit_dict[param] = [x / num_runs for x in rewards]

        return bandit_dict


def plot_bandits(reward_dict:dict[list[float]], 
                 bandit_type:str, 
                 show_plot:bool=False
                 ):
    """
    This function plots the average reward for each parameter.
    """

    if bandit_type == 'epsilon_greedy':
        plt.title('Epsilon Greedy Bandit Average Reward')
        title = '\epsilon'
    elif bandit_type == 'optimistic_initialization':
        plt.title('Optimistic Initialization Bandit Average Reward')
        title = 'Q_1'
    elif bandit_type == 'ucb':
        plt.title('UCB Bandit Average Reward')
        title = 'c'

    for key in reward_dict:
        plt.plot(reward_dict[key], label= f'${title} = ${key}')

    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig(f"../plots/{bandit_type}.png")

    if show_plot:
        plt.show()
    plt.close()

def get_best(d:dict[list[float]]):
    """
    This function returns the best performing parameter for the bandit.
    """
    max_avg = max(sum(values) / len(values) for values in d.values())
    for key, value in d.items():
        if sum(value) / len(value) == max_avg:
            return key

def plot_best(epsilon_greedy_dict:dict[list[float]],
               optimistic_greedy_dict:dict[list[float]], 
               ucb_dict:dict[list[float]],
               show_plot:bool=False
               ):
    """
    This function plots the best performing parameter for each bandit.
    """

    # Get best parameter for each bandit algorithm:
    best_eps = get_best(epsilon_greedy_dict)
    best_opt = get_best(optimistic_greedy_dict)
    best_ucb = get_best(ucb_dict)

    plt.plot(epsilon_greedy_dict[best_eps], label= f'$\epsilon =$ {best_eps}')
    plt.plot(optimistic_greedy_dict[best_opt], label= f'$Q_1 =$ {best_opt}')
    plt.plot(ucb_dict[best_ucb], label= f'$c =$ {best_ucb}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Best Multi-Armed Bandit Average Reward')
    plt.legend()
    plt.savefig("../plots/comparison.png")

    if show_plot:
        plt.show()
    plt.close()
