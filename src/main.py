import argparse

from epsilon_greedy import EpsilonGreedyBandit
from optimistic_initialization import OptimisticGreedyBandit
from ucb import UCBBandit
from utils import plot_bandits, plot_best, run_bandits

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations to run the bandit algorithms")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs to average over")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms in the bandit")
    parser.add_argument("--mu", type=float, default=0, help="Mean of the normal distribution")
    parser.add_argument("--var", type=float, default=3, help="Variance of the normal distribution")
    parser.add_argument("--show_plot", action="store_true", help="Whether to show the plot or not")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.1, 0.2], help="Epsilon values for the Epsilon Greedy Bandit algorithm")
    parser.add_argument("--Qs", nargs="+", type=float, default=[1, 2, 5], help="Q1 values for the Optimistic Greedy Bandit algorithm")
    parser.add_argument("--c", nargs="+", type=float, default=[0.5, 2, 4], help="c values for the UCB Bandit algorithm")

    args = vars(parser.parse_args())
    
    # Run each bandit algorithm for each parameter:
    epsilon_greedy_dict = run_bandits(args["num_iterations"], 
                                        args["num_runs"], 
                                        args["num_arms"],
                                        args["mu"],
                                        args["var"], 
                                        EpsilonGreedyBandit,
                                        params = args["eps"]
                                        )
    
    optimistic_greedy_dict = run_bandits(args["num_iterations"], 
                                            args["num_runs"], 
                                            args["num_arms"], 
                                            args["mu"], 
                                            args["var"], 
                                            OptimisticGreedyBandit,
                                            params = args["Qs"]
                                            )
    
    ucb_dict = run_bandits(args["num_iterations"], 
                            args["num_runs"], 
                            args["num_arms"],
                            args["mu"], 
                            args["var"], 
                            UCBBandit,
                            params = args["c"]
                        )

    #Create plots for each bandit algorithm:
    plot_bandits(epsilon_greedy_dict, "epsilon_greedy", show_plot=args["show_plot"])
    plot_bandits(optimistic_greedy_dict, "optimistic_initialization", show_plot=args["show_plot"])
    plot_bandits(ucb_dict, "ucb", show_plot=args["show_plot"])

    # Plot best bandit from all methods:
    plot_best(epsilon_greedy_dict, optimistic_greedy_dict, ucb_dict)

if __name__ == "__main__":
    main()
