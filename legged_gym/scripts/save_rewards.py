
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
env_cfg.env.num_envs = 10

env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.measure_heights = False
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False

env_cfg.noise.add_noise = False

env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False


experiment_num = 1
train_cfg.runner.load_run = args.experiment + str(experiment_num)

rewards = []


# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

#Deploy every policy (saved every 5 iterations)
for policy_iter in range(0, 1001, 50):

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = policy_iter
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    #Deploy policy over 10 envs at once
    #Compute total reward accross all envs.
    reward = 0
    for i in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        #reward += torch.sum(infos['non_RM_reward']).item()
        reward += torch.sum(rews).item()

    #Add avg reward a single policy achieved
    print(reward)
    rewards.append(reward/env_cfg.env.num_envs)

#Now append to file
file = open(args.experiment+'_rewards.txt', "a")
for r in rewards:
    file.write(str(r) + ' ')
file.write('\n')
file.close()