#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, alpha, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # TO DO: Initialize relevant elements
        self.alpha = alpha
        self.Q = np.zeros((self.n_states,self.n_actions)) # Q(s,a)
        self.count = np.zeros((self.n_states,self.n_actions, self.n_states)) # count(s,a,s')
        self.rSum = np.zeros((self.n_states,self.n_actions, self.n_states)) # Sum of each reward for each state to next state with action
        self.esti_prob = np.zeros((self.n_states,self.n_actions, self.n_states)) # P(s'|s,a)
        self.esti_r = np.zeros((self.n_states,self.n_actions, self.n_states)) # R(s,a,s')
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        random_value = np.random.random(1)[0]
        if random_value <= 1 - epsilon:
            return np.argmax(self.Q[s])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[s])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add Dyna update
        self.count[s,a, s_next] += 1
        self.rSum[s,a, s_next] += r
        
        self.esti_prob[s,a, s_next] = self.count[s,a, s_next] / np.sum(self.count[s,a,:])
        self.esti_r[s,a, s_next]  = self.rSum[s,a, s_next] / self.count[s,a, s_next]
        if done:
            self.Q[s,a] += self.alpha * (r - self.Q[s,a])
        else:
            self.Q[s,a] += self.alpha * (r + self.gamma * np.max(self.Q[s_next,:]) - self.Q[s,a])

        
        for plan in range(n_planning_updates):
            # state - plan
            visited_states = [s for s in range(self.count.shape[0]) if np.sum(self.count[s].flatten()) > 0]
            random_state = np.random.choice(visited_states)
            # action - plan
            visited_actions = [a for a in range(len(self.count[random_state])) if np.sum(self.count[random_state][a]) != 0]
            random_action = np.random.choice(visited_actions)
            
            # next state - plan
            sim_probs = self.count[random_state, random_action] / np.sum(self.count[random_state, random_action])
            next_state = np.random.choice(range(self.n_states), p=sim_probs)
            plannd_reward = self.esti_r[random_state, random_action, next_state]
            # Update Q-value
            self.Q[random_state, random_action] += self.alpha * (plannd_reward + self.gamma * np.max(self.Q[next_state]) - self.Q[random_state, random_action])    
    
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, alpha, epsilon, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements
        self.alpha = alpha
        self.Q = np.zeros((self.n_states,self.n_actions)) # Q(s,a)
        self.count = np.zeros((self.n_states,self.n_actions, self.n_states)) # count(s,a,s')
        self.rSum = np.zeros((self.n_states,self.n_actions, self.n_states)) # Sum of each reward for each state to next state with action
        self.esti_prob = np.zeros((self.n_states,self.n_actions, self.n_states)) # P(s'|s,a)
        self.esti_r = np.zeros((self.n_states,self.n_actions, self.n_states)) # R(s,a,s')
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        random_value = np.random.random(1)[0]
        if random_value <= 1 - epsilon:
            return np.argmax(self.Q[s])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[s])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        
        # TO DO: Add Prioritized Sweeping code
        
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue
        self.count[s,a, s_next] += 1
        self.rSum[s,a, s_next] += r
        self.esti_prob[s,a, s_next] = self.count[s,a, s_next] / np.sum(self.count[s,a,:])
        self.esti_r[s,a, s_next]  = self.rSum[s,a, s_next] / self.count[s,a, s_next]
        
        if done:
            priority = abs(r - self.Q[s,a])
        else:
            priority = abs(r + self.gamma * np.max(self.Q[s_next,:]) - self.Q[s,a])

        if priority > self.priority_cutoff:
            self.queue.put((-priority,(s,a)))

        # Start sampling from the PQ to perform updates
        for j in range(n_planning_updates):
            # pop the highest priority from PQ
            if self.queue.empty():
                break
            _,(s,a) = self.queue.get() # get the top (s,a) for the queue

            s_next_probs = self.esti_prob[s, a]
            if np.sum(s_next_probs) == 0:
                continue  # No model yet
            s_next = np.argmax(s_next_probs)
            
            r = self.esti_r[s, a, s_next]
            # Update Q-value
            self.Q[s,a] += self.alpha * (r + self.gamma * np.max(self.Q[s_next,:]) - self.Q[s,a])
            #Loop over all state action that may lead to state s
            visited_list = []
            for i in range(self.n_states):
                for k in range(self.n_actions):
                    if self.count[i,k,s] > 0 and (i,k) not in visited_list:
                        visited_list.append((i,k))
            for s_bar, a_bar in visited_list:
                # get reward from model 
                r = self.esti_r[s_bar, a_bar, s]
                # compute the priority
                priority = abs(r + self.gamma * np.max(self.Q[s,:]) - self.Q[s_bar,a_bar])  
                if priority > self.priority_cutoff:
                    self.queue.put((-priority,(s_bar,a_bar))) # put (s_bar,a_bar) on the queue with priority p
                    
                    
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q[s])  # ← greedy action (no epsilon)
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma,alpha = 0.5,epsilon = 0.1) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma, alpha = 0.5, epsilon = 0.1) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s, 0.1)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
