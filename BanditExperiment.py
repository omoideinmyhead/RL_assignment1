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
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    # Assignment 1: e-greedy
    egreedy = EgreedyPolicy(n_actions)
    '''
    rewards = np.zeros(n_timesteps)
    for i in range(n_timesteps):
        env = BanditEnvironment(n_actions)
        a = egreedy.select_action(0.01)
        r = env.act(a)
        egreedy.update(a, r)
        rewards[i] = r 
    '''
    rewards = np.zeros((n_repetitions, n_timesteps))
    for i in range(n_repetitions):
        env = BanditEnvironment(n_actions)
        for j in range(n_timesteps):
            a = egreedy.select_action(0.01)
            r = env.act(a)
            egreedy.update(a, r)
            rewards[i, j] = r

    # LearningCurvePlot.add_curve( , "Average rewards results")

    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)