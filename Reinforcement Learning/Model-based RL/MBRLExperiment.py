#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth
import time


def run_repetitions(n_repetitions, n_timesteps, eval_interval,
                    learning_rate, gamma, alpha, epsilon, 
                    n_planning_updates, wind_proportion, priority_cutoff=0.1,
                    agent_type='Dyna', max_episode_length=100,
                    n_eval_episodes=30, smoothing_window=5):
    
    eval_points = np.arange(0, n_timesteps + 1, eval_interval)
    all_returns = np.zeros((n_repetitions, len(eval_points)))

    for rep in range(n_repetitions):
        print(f"Repetition {rep+1}/{n_repetitions} | Wind: {wind_proportion}, Planning: {n_planning_updates}, Agent: {agent_type}")
        env = WindyGridworld(wind_proportion=wind_proportion)

        if agent_type == 'Dyna':
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma, alpha, epsilon)
        elif agent_type == 'PS':
            agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma, alpha, epsilon, priority_cutoff)
        else:
            raise NotImplementedError("No such agent")

        s = env.reset()
        eval_index = 0
        eval_env = WindyGridworld(wind_proportion=wind_proportion)
        all_returns[rep, eval_index] = agent.evaluate(eval_env, n_eval_episodes, max_episode_length)
        eval_index += 1

        for t in range(1, n_timesteps + 1):
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            s = env.reset() if done else s_next

            if t % eval_interval == 0:
                eval_return = agent.evaluate(eval_env, n_eval_episodes, max_episode_length)
                all_returns[rep, eval_index] = eval_return
                # print(f"  Step {t}: Eval return = {eval_return:.2f}")
                eval_index += 1

    mean_returns = np.mean(all_returns, axis=0)
    if smoothing_window > 0:
        smoothed_returns = smooth(mean_returns, window=smoothing_window)
    elif smoothing_window == 0:
        smoothed_returns = mean_returns

    mean_eval_return = np.mean(all_returns)
    # print(f"Average return over all evaluations for {agent_type} with n_planning_updates {n_planning_updates}: {mean_eval_return:.2f}")

    # print("\nMean return at each evaluation point:")
    # print("{:<12} {:<10}".format("Timestep", "Mean Return"))
    # print("-" * 24)
    # for t, ret in zip(eval_points, mean_returns):
    #     print("{:<12} {:<10.2f}".format(t, ret))

    return eval_points, smoothed_returns


def experiment(agent):
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updatess = [1,3,5] 
    
    agent = agent

    priority_cutoff = 0.01

    smoothing_window=5

    for wind in wind_proportions:
        # Create a new plot for each wind setting
        plot = LearningCurvePlot(title=f"{agent}Agent Learning Curve (Wind={wind})")

        # Create the baseline Q-learning curve
        x, y = run_repetitions(
                    n_repetitions=n_repetitions,
                    n_timesteps=n_timesteps,
                    eval_interval=eval_interval,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    alpha=learning_rate,
                    epsilon=epsilon,
                    n_planning_updates=0,  # 0 is Q-learning 
                    wind_proportion=wind,
                    priority_cutoff=priority_cutoff,
                    agent_type='Dyna',
                    max_episode_length=100,
                    n_eval_episodes=30,
                    smoothing_window=smoothing_window)

        plot.add_curve(x, y, label="Q-learning baseline")

        for planning_updates in n_planning_updatess:
            x, y = run_repetitions(
                    n_repetitions=n_repetitions,
                    n_timesteps=n_timesteps,
                    eval_interval=eval_interval,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    alpha=learning_rate,
                    epsilon=epsilon,
                    n_planning_updates=planning_updates,
                    wind_proportion=wind,
                    priority_cutoff=priority_cutoff,
                    agent_type=agent,
                    max_episode_length=100,
                    n_eval_episodes=30,
                    smoothing_window=smoothing_window 
            )

            label = f"Planning={planning_updates}"
            plot.add_curve(x, y, label=label)

        plot.set_ylim(-105, 105)

        # Save the plot for this wind proportion
        plot.save(f"{agent}_learning_curve_wind_{wind}.png")

def single_experiment(agent, wind, n_planning_updates):

    #Runs a single experiment and plots learning curve with timing info.
 
    # Hyperparameters
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 1
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    max_episode_length = 100
    n_eval_episodes = 30
    smoothing_window = 5
    priority_cutoff = 0.01  # Only used if agent is prioritized sweeping


    # Start timing
    start_time = time.time()

    # Run repetitions and collect results
    x, y = run_repetitions(
        n_repetitions=n_repetitions,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        gamma=gamma,
        alpha=learning_rate,
        epsilon=epsilon,
        n_planning_updates=n_planning_updates,
        wind_proportion=wind,
        priority_cutoff=priority_cutoff,
        agent_type=agent,
        max_episode_length=max_episode_length,
        n_eval_episodes=n_eval_episodes,
        smoothing_window=smoothing_window
    )

    # Stop timing
    end_time = time.time()
    duration = end_time - start_time
    print(f"{agent}, {n_planning_updates}, {wind}: Experiment completed in {duration:.2f} seconds")

    # Plot results
    plot = LearningCurvePlot(title=f"{agent} Agent (Wind={wind}, Planning={n_planning_updates})")
    label = f"Planning={n_planning_updates}"
    plot.add_curve(x, y, label=label)
    plot.set_ylim(-105, 105)
    plot.save(f"{agent}learning_curve_wind{wind}planning{n_planning_updates}.png")
   
def comparison_det():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[1.0]
    

    priority_cutoff = 0.01

    smoothing_window = 0

    for wind in wind_proportions:
        # Create a new plot for each wind setting
        plot = LearningCurvePlot(title=f"Comparison of Various Agents' Learning Curves (Wind={wind})")

        # baseline Q-learning curve
        agent = 'Dyna'
        x, y = run_repetitions(
                    n_repetitions=n_repetitions,
                    n_timesteps=n_timesteps,
                    eval_interval=eval_interval,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    alpha=learning_rate,
                    epsilon=epsilon,
                    n_planning_updates=0,  # 0 is Q-learning 
                    wind_proportion=wind,
                    priority_cutoff=priority_cutoff,
                    agent_type=agent,
                    max_episode_length=100,
                    n_eval_episodes=30,
                    smoothing_window=smoothing_window)

        plot.add_curve(x, y, label="Q-learning baseline")

        # Best Dyna agent
        agent = 'Dyna'
        planning_updates = 5                    # change this to the best one
        x, y = run_repetitions(
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                alpha=learning_rate,
                epsilon=epsilon,
                n_planning_updates=planning_updates,
                wind_proportion=wind,
                priority_cutoff=priority_cutoff,
                agent_type=agent,
                max_episode_length=100,
                n_eval_episodes=30,
                smoothing_window=smoothing_window 
        )

        label = f"{agent}Agent, Planning={planning_updates}"
        plot.add_curve(x, y, label=label)

        # Best PS agent
        agent = 'PS'
        planning_updates = 5                    # change this to the best one
        x, y = run_repetitions(
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                alpha=learning_rate,
                epsilon=epsilon,
                n_planning_updates=planning_updates,
                wind_proportion=wind,
                priority_cutoff=priority_cutoff,
                agent_type=agent,
                max_episode_length=100,
                n_eval_episodes=30,
                smoothing_window=smoothing_window 
        )

        label = f"{agent}Agent, Planning={planning_updates}"
        plot.add_curve(x, y, label=label)

        plot.set_ylim(-105, 105)

        # Save the plot for this wind proportion
        plot.save(f"Comparison_learning_curves_wind_{wind}.png")

def comparison_sto():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9]
    

    priority_cutoff = 0.01

    smoothing_window = 5

    for wind in wind_proportions:
        # Create a new plot for each wind setting
        plot = LearningCurvePlot(title=f"Comparison of Various Agents' Learning Curves (Wind={wind})")

        # baseline Q-learning curve
        agent = 'Dyna'
        x, y = run_repetitions(
                    n_repetitions=n_repetitions,
                    n_timesteps=n_timesteps,
                    eval_interval=eval_interval,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    alpha=learning_rate,
                    epsilon=epsilon,
                    n_planning_updates=0,  # 0 is Q-learning 
                    wind_proportion=wind,
                    priority_cutoff=priority_cutoff,
                    agent_type=agent,
                    max_episode_length=100,
                    n_eval_episodes=30,
                    smoothing_window=smoothing_window)

        plot.add_curve(x, y, label="Q-learning baseline")

        # Best Dyna agent
        agent = 'Dyna'
        planning_updates = 5                    # change this to the best one
        x, y = run_repetitions(
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                alpha=learning_rate,
                epsilon=epsilon,
                n_planning_updates=planning_updates,
                wind_proportion=wind,
                priority_cutoff=priority_cutoff,
                agent_type=agent,
                max_episode_length=100,
                n_eval_episodes=30,
                smoothing_window=smoothing_window 
        )

        label = f"{agent}Agent, Planning={planning_updates}"
        plot.add_curve(x, y, label=label)

        # Best PS agent
        agent = 'PS'
        planning_updates = 3                    # change this to the best one
        x, y = run_repetitions(
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                alpha=learning_rate,
                epsilon=epsilon,
                n_planning_updates=planning_updates,
                wind_proportion=wind,
                priority_cutoff=priority_cutoff,
                agent_type=agent,
                max_episode_length=100,
                n_eval_episodes=30,
                smoothing_window=smoothing_window 
        )

        label = f"{agent}Agent, Planning={planning_updates}"
        plot.add_curve(x, y, label=label)

        plot.set_ylim(-105, 105)

        # Save the plot for this wind proportion
        plot.save(f"Comparison_learning_curves_wind_{wind}.png")

def run_repetitions_adapt(n_repetitions, n_timesteps, eval_interval,
                    learning_rate, gamma, alpha, epsilon, 
                    n_planning_updates, wind_proportion_1, wind_proportion_2, priority_cutoff=0.1,
                    agent_type='Dyna', max_episode_length=100,
                    n_eval_episodes=30, smoothing_window=5):
    # for experiment where wind proportion changes halfway
    
    eval_points = np.arange(0, n_timesteps + 1, eval_interval)
    all_returns = np.zeros((n_repetitions, len(eval_points)))

    for rep in range(n_repetitions):
        wind = wind_proportion_1
        print(f"Repetition {rep+1}/{n_repetitions} | Wind: {wind_proportion_1} - {wind_proportion_2}, Planning: {n_planning_updates}, Agent: {agent_type}")
        env = WindyGridworld(wind_proportion=wind)

        if agent_type == 'Dyna':
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma, alpha, epsilon)
        elif agent_type == 'PS':
            agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma, alpha, epsilon, priority_cutoff)
        else:
            raise NotImplementedError("No such agent")

        s = env.reset()
        eval_index = 0
        eval_env = WindyGridworld(wind_proportion=wind)
        all_returns[rep, eval_index] = agent.evaluate(eval_env, n_eval_episodes, max_episode_length)
        eval_index += 1

        for t in range(1, n_timesteps + 1):
            if t == 5000:
                env.set_wind_proportion(wind_proportion=wind_proportion_2)
                eval_env.set_wind_proportion(wind_proportion=wind_proportion_2)
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            s = env.reset() if done else s_next

            if t % eval_interval == 0:
                eval_return = agent.evaluate(eval_env, n_eval_episodes, max_episode_length)
                all_returns[rep, eval_index] = eval_return
                # print(f"  Step {t}: Eval return = {eval_return:.2f}")
                eval_index += 1

    mean_returns = np.mean(all_returns, axis=0)
    if smoothing_window > 0:
        smoothed_returns = smooth(mean_returns, window=smoothing_window)
    elif smoothing_window == 0:
        smoothed_returns = mean_returns

    mean_eval_return = np.mean(all_returns)
    # print(f"Average return over all evaluations for {agent_type} with n_planning_updates {n_planning_updates}: {mean_eval_return:.2f}")

    # print("\nMean return at each evaluation point:")
    # print("{:<12} {:<10}".format("Timestep", "Mean Return"))
    # print("-" * 24)
    # for t, ret in zip(eval_points, mean_returns):
    #     print("{:<12} {:<10.2f}".format(t, ret))

    return eval_points, smoothed_returns
    
def comparison_adapt():
    # for changing the wind proportion at 5000 timesteps

    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportion_1 = 0.5
    wind_proportion_2 = 0.5

    priority_cutoff = 0.01

    smoothing_window = 0

    
    # Create a new plot for each wind setting
    plot = LearningCurvePlot(title=f"Comparison of Various Agents' Learning Curves (Wind={wind_proportion_1} - {wind_proportion_2})")

    # baseline Q-learning curve
    agent = 'Dyna'
    x, y = run_repetitions_adapt(
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                alpha=learning_rate,
                epsilon=epsilon,
                n_planning_updates=0,  # 0 is Q-learning 
                wind_proportion_1=wind_proportion_1,
                wind_proportion_2=wind_proportion_2,
                priority_cutoff=priority_cutoff,
                agent_type=agent,
                max_episode_length=100,
                n_eval_episodes=30,
                smoothing_window=smoothing_window)

    plot.add_curve(x, y, label="Q-learning baseline")

    
    agent = 'PS'
    planning_updates = 1                    
    x, y = run_repetitions_adapt(
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=learning_rate,
            epsilon=epsilon,
            n_planning_updates=planning_updates,
            wind_proportion_1=wind_proportion_1,
            wind_proportion_2=wind_proportion_2,
            priority_cutoff=priority_cutoff,
            agent_type=agent,
            max_episode_length=100,
            n_eval_episodes=30,
            smoothing_window=smoothing_window 
    )

    label = f"{agent}Agent, Planning={planning_updates}"
    plot.add_curve(x, y, label=label)

    
    agent = 'PS'
    planning_updates = 3               
    x, y = run_repetitions_adapt(
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=learning_rate,
            epsilon=epsilon,
            n_planning_updates=planning_updates,
            wind_proportion_1=wind_proportion_1,
            wind_proportion_2=wind_proportion_2,
            priority_cutoff=priority_cutoff,
            agent_type=agent,
            max_episode_length=100,
            n_eval_episodes=30,
            smoothing_window=smoothing_window 
    )

    label = f"{agent}Agent, Planning={planning_updates}"
    plot.add_curve(x, y, label=label)

    
    agent = 'PS'
    planning_updates = 5                  
    x, y = run_repetitions_adapt(
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=learning_rate,
            epsilon=epsilon,
            n_planning_updates=planning_updates,
            wind_proportion_1=wind_proportion_1,
            wind_proportion_2=wind_proportion_2,
            priority_cutoff=priority_cutoff,
            agent_type=agent,
            max_episode_length=100,
            n_eval_episodes=30,
            smoothing_window=smoothing_window 
    )

    label = f"{agent}Agent, Planning={planning_updates}"
    plot.add_curve(x, y, label=label)

    plot.set_ylim(-105, 105)

    # Save the plot for this wind proportion
    plot.save(f"Comparison_learning_curves_wind_{wind_proportion_1}-{wind_proportion_2}.png")

if __name__ == '__main__':
    experiment('Dyna')
    experiment('PS')
    # single_experiment('Dyna', 0.9, 1)
    comparison_det()
    comparison_sto()

    # ##running single experiments to get the run time data
    agents_planning = {
        'Dyna': [0, 1, 3, 5],
        'PS': [1, 3, 5]
    }

    winds = [0.9, 1.0]

    for wind in winds:
        for agent, planning_list in agents_planning.items():
            for n_planning_updates in planning_list:
                single_experiment(agent, wind, n_planning_updates)

    comparison_adapt()

