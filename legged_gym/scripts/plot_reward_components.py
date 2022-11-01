
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt

labels = ['action_rate', 'ang_vel_xy', 'base_height', 
        'collision', 'dof_acc', 'dof_pos_limits', 
        'feet_air_time', 'lin_vel_z', 'orientation', 
        'torques', 'tracking_ang_vel', 'tracking_lin_vel']

all_reward_components = []

#Read rewards file per each experiment type
file = open('rm_pace_reward_components.txt', "r")
file_lines = file.readlines()


for line in file_lines:

    rewards = line.split(' ')[1:-1]
    rewards = [float(r) for r in rewards]
    all_reward_components.append(rewards)


fig, ax = plt.subplots()

#Each iter has 24*4096 steps. We plot on x-axis every 50 iters
#So a new point on x-axis correspond to 24*4096*50 = 4,915,200 more training iters
#Lets call it 5 million
time = [i*5 for i in range(0, len(all_reward_components[0]))]

for i, component in enumerate(all_reward_components):

    ax.plot(time, component, label=labels[i])

ax.legend()

plt.xlabel('Training Steps (in millions)')
plt.ylabel('Reward')
plt.savefig("reward_components.pdf", bbox_inches='tight')
