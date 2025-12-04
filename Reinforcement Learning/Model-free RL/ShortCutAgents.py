import copy
import numpy as np
import ShortCutEnvironment as c

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        

    def select_action(self, state):
        # TO DO: Implement policy

        #Implement an ϵ-greedy policy for selecting an action.
        random_value = np.random.random(1)[0]
        if random_value <= 1 - self.epsilon:
            return np.argmax(self.Q[state])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[state])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        # action = None
        # return action

        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        if not done:
            self.Q[state][action] += self.alpha*(reward + self.gamma* np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha*(reward - self.Q[state][action])
        
    
    def train(self, n_episodes, env,n_timesteps):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset()  
            total_reward = 0
            while not env.done():
                state = env.state()
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()

                self.update(state, action, reward, next_state, done)

                total_reward += reward

            episode_returns.append(total_reward)
        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # TO DO: Implement policy
        random_value = np.random.random(1)[0]
        if random_value <= 1 - self.epsilon:
            return np.argmax(self.Q[state])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[state])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        
    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        if not done:
            self.Q[state][action] += self.alpha*(reward + self.gamma* self.Q[next_state][next_action] - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha*(reward - self.Q[state][action])
        

    def train(self, n_episodes, env, n_timesteps):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset() 
            total_reward = 0
            state = env.state()  
            action = self.select_action(state)  

            while not env.done():
                reward = env.step(action)
                next_state = env.state()
                done = env.done()

                next_action = self.select_action(next_state) 

    
                self.update(state, action, reward, next_state, next_action, done)

                total_reward += reward
                state = next_state  
                action = next_action  

            episode_returns.append(total_reward)

        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        random_value = np.random.random(1)[0]
        if random_value <= 1 - self.epsilon:
            return np.argmax(self.Q[state])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[state])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        if not done:
            expected_Q_value = 0
            for next_action in range(self.n_actions):
                # Calculate the expected Q-value for the next state
                if next_action == np.argmax(self.Q[next_state]):
                    # for the action with the highest Q-value, the probability is (1 - epsilon)
                    expected_Q_value += (1 - self.epsilon) * self.Q[next_state][next_action]
                else:
                    # for all other actions, the probability is (epsilon / (n_actions - 1))
                    expected_Q_value += (self.epsilon / (self.n_actions - 1)) * self.Q[next_state][next_action]
            # Update the Q-value for the current state-action pair
            self.Q[state][action] += self.alpha*(reward + self.gamma*expected_Q_value - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha*(reward - self.Q[state][action])

    def train(self, n_episodes, env, n_timesteps):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset() 
            total_reward = 0
            step_count = 0
            state = env.state()  
            action = self.select_action(state) 

            while not env.done():
                reward = env.step(action)
                next_state = env.state()
                done = env.done()

                self.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                action = self.select_action(state) 
                step_count += 1

            episode_returns.append(total_reward)

        return episode_returns


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n # n-step SARSA
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # TO DO: Implement policy
        random_value = np.random.random(1)[0]
        if random_value <= 1 - self.epsilon:
            return np.argmax(self.Q[state])
        else:
            tempActions = list(range(0,self.n_actions))
            forbidden_index = np.argmax(self.Q[state])
            tempActions.pop(forbidden_index)
            action = np.random.choice(tempActions)
            return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        time_step = len(states)
        G = 0
        # Calculate the return G for the n-step SARSA update，start from the last time step
        # First use for loop to calculate the G value
        for i in range(time_step):
            G += (self.gamma**i) * rewards[i]

        if not done:
            # Update the Q-value for the state-action pair
            self.Q[states[0]][actions[0]] += self.alpha * (G + (np.power(self.gamma, time_step-1)* self.Q[states[time_step - 1]][actions[time_step - 1]]) - self.Q[states[0]][actions[0]])
        else:
            self.Q[states[0]][actions[0]] += self.alpha * (G - self.Q[states[0]][actions[0]])

    def train(self, n_episodes, env, n_timesteps):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        step_count = 0
        for _ in range(n_episodes):
            env.reset()  
            total_reward = 0
            states = []
            actions = []
            rewards = []
            state = env.state()  
            action = self.select_action(state)  
            step_count+= 1

            # Take n+1 actions before performing the first update
            for _ in range(self.n + 1):
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                if done:
                    break
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                state = next_state
                action = self.select_action(state)

            self.update(states, actions, rewards, done)
            states.pop(0)
            actions.pop(0)
            rewards.pop(0)

            # Continue taking actions and updating the Q-values
            while not env.done():
                reward = env.step(action)
                next_state = env.state()
                done = env.done()

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                self.update(states, actions, rewards, done)
                # Remove the oldest state-action-reward triplet from the list
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

                total_reward += reward
                state = next_state  
                action = self.select_action(state)

            # If there are any remaining states, actions, and rewards, update the Q-values
            if len(states) > 0:
                self.update(states, actions, rewards, done)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            env.reset()  
            episode_returns.append(total_reward)
        return episode_returns  
    
    
