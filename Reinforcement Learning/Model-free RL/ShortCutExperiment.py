import ShortCutEnvironment as se
import ShortCutAgents as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#@title Code Initialization
#@markdown <u>Please run this cell block before continuing. This code contains some plotting functions you will use throughout this notebook. You can inspect the code, but it is not required to fully understand it.</u>

# Plotting helper code -- run this cell before continuing.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')
        self.ax.set_ylim([-500,0])
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
            self.ax.legend()
        else:
            self.ax.plot(y)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


def run_repetitions(
        plot: LearningCurvePlot,
        env_class,
        agent_class,
        label,
        alpha,
        epsilon,
        n_rep,
        n_episodes,
        n_timesteps,
        n = 5,
        seed=42,
        **kwargs
    ):
    """
        Performs an experiment with a given agent class by
        repeatedly training a new agent instance on a new
        environment instance. Returns the average rewards
        over all repititions.
    """
    np.random.seed(seed) # For reproducibility

    n_repetitions = n_rep # Repeat experiment n_rep times to eliminate randomness
    n_timesteps = n_timesteps # We give the agent n_timesteps "learning" steps per repetition

    rewards = np.zeros((n_repetitions, n_episodes))
    if agent_class == sa.nStepSARSAAgent: label = f"n={n}"
    for rep in range(n_repetitions):
        # New environment and agent every repetition !!
        env = env_class()
        if agent_class == sa.nStepSARSAAgent:
            agent = agent_class(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha, epsilon=epsilon, n = n)
            
        else:
            agent = agent_class(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha, epsilon=epsilon)
        rewards[rep] = agent.train(n_episodes=n_episodes, env=env, n_timesteps=n_timesteps)
        # print(rewards[rep])
        
    average_rewards = np.mean(rewards, axis=0) # average over the repetitions
    # last_episodes_mean =  np.mean(average_rewards[900:])
    # print(alpha, " ", last_episodes_mean)
    reward_curve = smooth(average_rewards, window =50)
    plot.add_curve(reward_curve, label=label)

def single_experiment(
        env_class,
        agent_class,
        n_episodes =10000,
        alpha =0.1,
        epsilon =0.1,
        p = None,
        n = 5,
        n_timesteps =500,
        **kwargs
    ):
    """
        Performs an experiment with a given agent class by
        repeatedly training a new agent instance on a new 
        environment instance. Returns the average rewards
        over all repititions.
    """
    if p is not None:
        env = env_class()
        env.set_wind_probs(p)
    else:
        env = env_class()
    if agent_class == sa.nStepSARSAAgent:
        agent = agent_class(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha, epsilon=epsilon, n = n)
    else:
        agent = agent_class(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha, epsilon=epsilon)    
    agent.train(n_episodes=n_episodes, env=env, n_timesteps= n_timesteps)
    env.render_greedy(agent.Q) # Render the environment with the learned policy


if __name__ == "__main__":
    # ##------------------- Part 1: Q-Learning agent ----------------------------
   
    # ##------ 1b:
    alpha = 0.1
    epsilon = 0.1

    # ## Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for Q-learning agent...")
    # single_experiment(
    #     env_class=se.ShortcutEnvironment,
    #     agent_class=sa.QLearningAgent,
    #     n_episodes=10000
    # )

    # ## Run n_rep=100 repetitions of a similar experiment, but with n_episodes=1000 episodes
    # plot = LearningCurvePlot("QLearningAgent")
    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.QLearningAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon = epsilon,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 500,
    #     )
    # plt.show()

    # ## ------ 1c:
    # plot = LearningCurvePlot("QLearningAgent")
    # alpha_values = [0.01, 0.1, 0.5, 0.9]
    # for alpha in alpha_values:
    #     run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.QLearningAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    #     )
    # plt.show()

    # ##------------------- Part 2: SARSA agent ----------------------------
    # ##------ 2b:
    alpha = 0.1
    epsilon = 0.1

    ##Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for SARSA agent...")
    # single_experiment(
    #     env_class=se.ShortcutEnvironment,
    #     agent_class=sa.SARSAAgent,
    #     n_episodes=10000
    # )

    # ##Run n_rep=100 repetitions of a similar experiment, but with n_episodes=1000 episodes
    # plot = LearningCurvePlot("SARSAAgent")

    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.SARSAAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon = epsilon,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 500,
    #     )
    # plt.show()

    # ##------ 2c:
    # plot = LearningCurvePlot("SARSAAgent")

    # alpha_values = [0.01, 0.1, 0.5, 0.9]
    # for alpha in alpha_values:
    #     run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.SARSAAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    #     )
    # plt.show()

    # ##------------------- Part 3: Stormy Weather ----------------------------
    ## Q-learning Agent

    # alpha = 0.1
    # epsilon = 0.1

    # Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for Q-learning agent in Windy environment...")
    # single_experiment(
    #     env_class=se.WindyShortcutEnvironment,
    #     agent_class=sa.QLearningAgent,
    #     n_episodes=10000
    # )

    # # SARSA Agent
    # print("Running single experiment for SARSA agent in Windy environment...")
    # single_experiment(
    #     env_class=se.WindyShortcutEnvironment,
    #     agent_class=sa.SARSAAgent,
    #     n_episodes=10000
    # )


    # ##------------------- Part 4: Expected SARSA agent ----------------------------
    # ##------ 4b:
    # alpha = 0.1
    # epsilon = 0.1

    # ##Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for Expected SARSA agent...")
    # single_experiment(
    #     env_class=se.ShortcutEnvironment,
    #     agent_class=sa.ExpectedSARSAAgent,
    #     n_episodes=10000
    # )


    

    # ##------ 4c:
    # plot = LearningCurvePlot("Expected SARSA Agent")

    # alpha_values = [0.01, 0.1, 0.5, 0.9]
    # for alpha in alpha_values:
    #     run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.ExpectedSARSAAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    #     )
    # plt.show()

    # ##------------------- Part 5: n-step SARSA agent ----------------------------
    # ##------ 5b:
    alpha = 0.1
    epsilon = 0.1
    # # ##Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for n-step SARSA agent...")
    # single_experiment(
    #     env_class=se.ShortcutEnvironment,
    #     agent_class=sa.nStepSARSAAgent,
    #     n_episodes=10000,
    #     n = 1
    # )
    # ##------ 5c:
    # plot = LearningCurvePlot("n-step SARSA Agent")

    # n_values = [1, 2, 5, 10, 25]
    # for n in n_values:        #TODO not finished, this n needs implementing
    #     run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.nStepSARSAAgent,
    #         label=f"{chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n = n,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    #     )
    # plt.show()


# ## ------------ Comparison ---------------
    # plot = LearningCurvePlot("Comparing average rewards of different agents")
    # n=2
    # alpha=0.1
    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.nStepSARSAAgent,
    #         label=f"2-step SARSA",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n = n,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    # )
    # alpha=0.9
    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.ExpectedSARSAAgent,
    #         label=f"Expected SARSA {chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    # )
    # alpha=0.1
    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.SARSAAgent,
    #         label=f"SARSA {chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    # )
    # alpha=0.1
    # run_repetitions(
    #         plot,
    #         env_class=se.ShortcutEnvironment,
    #         agent_class=sa.QLearningAgent,
    #         label=f"Q-learning {chr(945)}: {alpha}",
    #         alpha=alpha,
    #         epsilon=0.1,
    #         n_rep = 100,
    #         n_episodes = 1000,
    #         n_timesteps = 200,
    # )
    # plt.show()

    # ##------------------- Part 6: change the chance of wind and observe the difference----------------------------
    
    probability = [0.1, 0.3, 0.5, 0.7]
    for p in probability:
        print(f"Running single experiment for SARSA agent in Windy environment with wind probability {p}")
        # plot = LearningCurvePlot(f"SARSA with wind probability {p}")
        single_experiment(
            env_class=se.differenceWindyShortEnvironment,
            agent_class=sa.SARSAAgent,
            n_episodes=10000,
            p = p
        )
    # plt.show()



    # Run a single experiment for n_episodes=10000 episodes
    # print("Running single experiment for Q-learning agent in Windy environment...")
    # single_experiment(
    #     env_class=se.WindyShortcutEnvironment,
    #     agent_class=sa.QLearningAgent,
    #     n_episodes=10000
    # )