
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt

args = get_args()

#experiments = ['naive', 'naive3T', 'augmented', 'rm']
experiments = ['augmented', 'rm']

all_rewards = []
all_stds = []

for experiment in experiments:

    #Read rewards file per each experiment type
    file = open(experiment + '_' + args.gait + '_rewards.txt', "r")
    file_lines = file.readlines()

    temp_rewards_storage = []

    for line in file_lines:

        rewards = line.split(' ')[:-1]
        rewards = [float(r) for r in rewards]
        temp_rewards_storage.append(rewards)

    #Compute average rewards and stds per experiment
    avg_experiment_rewards = []
    experiment_stds = []

    #Loop through all training iters
    for i in range(len(temp_rewards_storage[0])):

        #Store rewards from iter i at each experiment
        reward_i = []

        #Loop through all experiment runs
        for j in range(len(temp_rewards_storage)):

            reward_i.append(temp_rewards_storage[j][i])

        avg_experiment_rewards.append(np.mean(reward_i))
        experiment_stds.append(np.std(reward_i))

    all_rewards.append(avg_experiment_rewards)
    all_stds.append(experiment_stds)

all_stds = np.array(all_stds)

fig, ax = plt.subplots()
time = [i for i in range(0, len(all_rewards[0]))]

for i, experiment in enumerate(experiments):

    std_below = []
    std_above = []

    for t, reward in enumerate(all_rewards[i]):
        std_below.append(reward - all_stds[i,t])
        std_above.append(reward + all_stds[i,t])


    ax.plot(time, all_rewards[i], label=experiment)
    ax.fill_between(time, std_below, std_above, alpha=.1)

ax.legend()

plt.xlabel('Training Iterations')
plt.ylabel('Reward')
plt.savefig("plot.pdf", bbox_inches='tight')


"""for i,method in enumerate(methods):

    std_below = []
    std_above = []

    for t, reward in enumerate(rewards[i]):
        std_below.append(reward - reward_stds[i,t])
        std_above.append(reward + reward_stds[i,t])


    ax.plot(time, rewards[i], label=l)
    ax.fill_between(time, std_below, std_above, alpha=.1)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=13)

filename = version + '_reward_plot.pdf'

plt.xlabel('Training Iterations (in millions)')
plt.ylabel('Average Reward')

plt.yticks([x*100000 for x in [-1,0,1,2,3,4]], [-1, 0, 1, 2, 3, 4])
plt.xticks([x*10 for x in range(6)], [0, 1, 2, 3, 4, 5])

plt.savefig(filename, bbox_inches='tight')"""
