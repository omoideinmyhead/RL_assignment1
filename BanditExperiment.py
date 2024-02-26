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
 

def egreedy_experiment(epsilon, n_actions, n_timesteps, n_repetitions, smoothing_window):
    '''
    rewards = np.zeros(n_timesteps)
    egreedy = EgreedyPolicy(n_actions)
    env = BanditEnvironment(n_actions)
    for i in range(n_timesteps):
        a = egreedy.select_action(epsilon)
        r = env.act(a)
        egreedy.update(a, r)
        rewards[i] = r 
    '''
    rewards = np.zeros((n_repetitions, n_timesteps))
    rewards_total = np.zeros(n_timesteps)
    for i in range(n_repetitions):
        egreedy = EgreedyPolicy(n_actions)
        env = BanditEnvironment(n_actions)
        for j in range(n_timesteps):
            a = egreedy.select_action(epsilon)
            r = env.act(a)
            egreedy.update(a, r)
            rewards[i, j] = r
            rewards_total[j] += r
    avg_rewards = np.divide(rewards_total, n_repetitions)
    smoothed_rewards = smooth(avg_rewards, smoothing_window, poly=1)
    accumulated_rewards = sum(rewards_total)
    return smoothed_rewards, accumulated_rewards

def OI_experiment(initial_value, n_actions, n_timesteps, n_repetitions, smoothing_window):
    rewards = np.zeros((n_repetitions, n_timesteps))
    rewards_total = np.zeros(n_timesteps)
    for i in range(n_repetitions):
        oi = OIPolicy(n_actions, initial_value, learning_rate=0.1)
        env = BanditEnvironment(n_actions)
        for j in range(n_timesteps):
            a = oi.select_action()
            r = env.act(a)
            oi.update(a, r)
            rewards[i, j] = r
            rewards_total[j] += r
    avg_rewards = np.divide(rewards_total, n_repetitions)
    smoothed_rewards = smooth(avg_rewards, smoothing_window, poly=1)
    accumulated_rewards = sum(rewards_total)
    return smoothed_rewards, accumulated_rewards

def UCB_experiment(c, n_actions, n_timesteps, n_repetitions, smoothing_window):
    rewards = np.zeros((n_repetitions, n_timesteps))
    rewards_total = np.zeros(n_timesteps)
    for i in range(n_repetitions):
        env = BanditEnvironment(n_actions)
        ucb = UCBPolicy(n_actions)
        for j in range(n_timesteps):
            a = ucb.select_action(c, t=j)
            r = env.act(a)
            ucb.update(a, r)
            rewards[i, j] = r
            rewards_total[j] += r
    avg_rewards = np.divide(rewards_total, n_repetitions)
    smoothed_rewards = smooth(avg_rewards, smoothing_window, poly=1)
    accumulated_rewards = sum(rewards_total)
    return smoothed_rewards, accumulated_rewards


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    comparison = ComparisonPlot(title="Comparison")
    # egreedy
    epsilons = [0.01, 0.05, 0.1, 0.25]
    LC_egreedy = LearningCurvePlot(title="Learning Curve eGreedyPolicy")
    accumulated_rewards = np.zeros(4)
    for i in range(len(epsilons)):
        rewards,accumulated_reward = egreedy_experiment(epsilons[i], n_actions=n_actions,n_timesteps=n_timesteps,
                   n_repetitions=n_repetitions,smoothing_window=smoothing_window)
        x = np.arange(1000)
        y = rewards[x]
        accumulated_rewards[i] += accumulated_reward
        LC_egreedy.add_curve(y, label=f'epsilon={epsilons[i]}')
    x1 =  epsilons #[np.log(j) for j in epsilons]
    y1 = np.divide(accumulated_rewards,(n_repetitions * n_timesteps))
    comparison.add_curve(x1, y1, label='e-Greedy')
    LC_egreedy.save(name='eGreedy.png')

    # OIPolicy
    initial_values = [0.1, 0.5, 1.0, 2.0]
    LC_oi = LearningCurvePlot(title="Learning Curve OIPolicy")
    accumulated_rewards = np.zeros(4)
    for i in range(len(initial_values)):
        rewards, accumulated_reward = OI_experiment(initial_values[i], n_actions=n_actions,n_timesteps=n_timesteps,
                   n_repetitions=n_repetitions,smoothing_window=smoothing_window)
        x = np.arange(1000)
        y = rewards[x]
        accumulated_rewards[i] += accumulated_reward
        LC_oi.add_curve(y, label=f'initial value={initial_values[i]}')
    x2 = initial_values #[np.log(j) for j in initial_values]
    y2 = np.divide(accumulated_rewards,(n_repetitions * n_timesteps))
    comparison.add_curve(x2, y2, label='UCB')
    LC_oi.save(name='OI.png')

    # UCBPolicy
    c_values = [0.01,0.05,0.1,0.25,0.5,1.0]
    LC_ucb = LearningCurvePlot(title="Learning Curve UCBPolicy")
    accumulated_rewards = np.zeros(6)
    for i in range(len(c_values)):
        rewards, accumulated_reward = OI_experiment(c_values[i], n_actions=n_actions,n_timesteps=n_timesteps,
                   n_repetitions=n_repetitions,smoothing_window=smoothing_window)
        x = np.arange(1000)
        y = rewards[x]
        accumulated_rewards[i] += accumulated_reward
        LC_ucb.add_curve(y, label=f'c={c_values[i]}')
    x3 = c_values #[np.log(j) for j in c_values]
    y3 = np.divide(accumulated_rewards,(n_repetitions * n_timesteps))
    comparison.add_curve(x3, y3, label='greedy with optimal initialization, α = 0.1')
    LC_ucb.save(name='UCB.png')

    # Comparison
    comparison.save(name='comparison.png')