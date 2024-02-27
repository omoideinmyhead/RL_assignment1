#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initialize the mean values and counts for each action to 0
        self.Q = np.zeros(n_actions)
        self.counts = np.zeros(n_actions, dtype=int)
        
    def select_action(self, epsilon):
        # TO DO: Add own code
        if np.random.random() < epsilon:
            a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        else:
            a = np.argmax(self.Q)
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        self.counts[a] += 1
        self.Q[a] = self.Q[a] + (1/self.counts[a])*(r-self.Q[a])

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        # Initialize the estimates for the mean Q(a) of each arm to initial_value
        self.Q = [initial_value]*n_actions
        self.learning_rate = learning_rate
        
    def select_action(self):
        # TO DO: Add own code
        a = np.argmax(self.Q) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        self.Q[a] = self.Q[a] + self.learning_rate*(r-self.Q[a])

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # Initialize a vector with means Q(a) and counts n(a) of 0 for each action
        self.Q = np.zeros(n_actions)
        self.counts = np.zeros(n_actions, dtype=int)
    
    def select_action(self, c, t):
        # TO DO: Add own code
        a = np.argmax(self.Q + c*np.sqrt(np.log(t)/(self.counts+0.000001))) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        self.counts[a] += 1
        self.Q[a] = self.Q[a] + (1/self.counts[a])*(r-self.Q[a])
    '''
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.q=np.zeros(n_actions) #the means Q(a)
        self.n=np.zeros(n_actions) #counts n(a)

    def select_action(self, c, t):
        action_values = []
        for a in range(self.n_actions):
            if self.n[a] == 0:
                action_values.append(float('inf'))
            else:
                exploration_value = c * np.sqrt(np.log(t) / self.n[a])
                action_values.append(self.q[a] + exploration_value)
        return np.argmax(action_values)



    def update(self,a,r):

        #n(a)=n(a)+1
        self.n[a]= self.n[a]+1

        #Q(a)=Q(a)+ 1/(n(a)) [r(a)-Q(a)]
        self.q[a]= self.q[a] + (1/self.n[a]) * (r-self.q[a]) '''

    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
