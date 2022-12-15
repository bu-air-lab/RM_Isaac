
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import matplotlib.pyplot as plt


args = get_args()
args.headless = True
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

# override some parameters for testing
env_cfg.env.num_envs = 100

env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.measure_heights = False
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False

env_cfg.noise.add_noise = False

env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False

isNo_gait = False

if(args.experiment == 'no_gait'):
    train_cfg.runner.load_run = args.experiment + '_no_gait' + str(args.seed)
    isNo_gait = True

    #We evaluate no_gait on same reward as other baselines. naive is only baseline with same state space
    args.experiment = 'naive'
else:
    train_cfg.runner.load_run = args.experiment + '_' + args.gait + str(args.seed)


rewards = []
reward_components = {}
reward_components_iter = 0

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

#logger = Logger(env.dt)

#Deploy every policy (saved every 50 iterations)
for policy_iter in range(0, 1001, 50):

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = policy_iter
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    #Deploy policy over 100 envs at once
    #Compute total reward accross all envs.
    #Also keep track of all reward components
    reward = 0
    #added_components = 0
    for i in range(int(env.max_episode_length)):

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        reward += torch.sum(rews).item()

        step_reward_components = infos['reward_components']

        for key in step_reward_components.keys():
            if(key not in reward_components.keys()):
                reward_components[key] = [torch.sum(step_reward_components[key]).item()]
            elif(len(reward_components[key]) == reward_components_iter):
                reward_components[key].append(torch.sum(step_reward_components[key]).item())
            else:
                reward_components[key][reward_components_iter] += torch.sum(step_reward_components[key]).item()

            #added_components += torch.sum(step_reward_components[key]).item()

    #print("Reward:", reward)
    #print("Added components:", added_components)

    #Add avg reward a single policy achieved
    rewards.append(reward/env_cfg.env.num_envs)

    for key in reward_components.keys():
        reward_components[key][reward_components_iter] /= env_cfg.env.num_envs

    reward_components_iter += 1

    print(reward/env_cfg.env.num_envs)
    #print(reward_components)

#Now append to file
if(isNo_gait):
    args.experiment = 'no_gait'
file = open(args.experiment + '_' + args.gait + '_rewards.txt', "a")
for r in rewards:
    file.write(str(r) + ' ')
file.write('\n')
file.close()

# file = open(args.experiment + '_' + args.gait + '_reward_components.txt', "a")
# for key in reward_components.keys():
#     file.write(key + ': ')
#     for value in reward_components[key]:
#         file.write(str(value) + ' ')
#     file.write('\n')
# file.write('\n')
# file.close()