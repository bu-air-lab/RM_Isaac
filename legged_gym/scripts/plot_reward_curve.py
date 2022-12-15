
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt

#Dont use type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

args = get_args()

experiments = ['no_gait', 'naive', 'naive3T', 'augmented', 'rm']
#experiments = ['augmented', 'rm']
#experiments = ['rm']

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

#Each iter has 24*4096 steps. We plot on x-axis every 50 iters
#So a new point on x-axis correspond to 24*4096*50 = 4,915,200 more training iters
#Lets call it 5 million
time = [i*5 for i in range(0, len(all_rewards[0]))]

for i, experiment in enumerate(experiments):

    std_below = []
    std_above = []

    for t, reward in enumerate(all_rewards[i]):
        std_below.append(reward - all_stds[i,t])
        std_above.append(reward + all_stds[i,t])

    if(experiment == 'naive'):
        experiment = 'Naive'
    elif(experiment == 'naive3T'):
        experiment = 'Naive-3T'
    elif(experiment == 'augmented'):
        experiment = 'Augmented'
    elif(experiment == 'rm'):
        experiment = 'RMLL'
    elif(experiment == 'no_gait'):
        experiment = 'No-Gait'

    ax.plot(time, all_rewards[i], label=experiment)
    ax.fill_between(time, std_below, std_above, alpha=.1)

ax.legend()

if(args.gait == 'bound_walk'):
    args.gait = 'Three-One'

plt.xlabel('Training Steps (in millions)')
plt.ylabel('Reward')
plt.title('Reward Curves for ' + args.gait.capitalize() + ' Gait')
plt.savefig("plot_" + args.gait + ".pdf", bbox_inches='tight')
