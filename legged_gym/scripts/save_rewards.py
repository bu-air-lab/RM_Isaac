
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

train_cfg.runner.load_run = args.experiment + '_' + args.gait + str(args.seed)

rewards = []

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

#Deploy every policy (saved every 50 iterations)
for policy_iter in range(0, 1001, 50):

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = policy_iter
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    #policy = ppo_runner.get_inference_policy(device=env.device)
    policy, state_estimator = ppo_runner.get_inference_policy(device=env.device)

    #Deploy policy over 100 envs at once
    #Compute total reward accross all envs.
    reward = 0
    for i in range(int(env.max_episode_length)):

        #Estimate base vel and foot heights
        #noRM_history baseline gets access to ground truth instead
        if(args.experiment != 'noRM_history'):

            estimated_state = state_estimator(obs)
            obs = torch.cat((obs[:, :-7], estimated_state),dim=-1)

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        reward += torch.sum(rews).item()

    #Add avg reward a single policy achieved
    rewards.append(reward/env_cfg.env.num_envs)



    print(reward/env_cfg.env.num_envs)

#Now append to file
file = open(args.experiment + '_' + args.gait + '_rewards.txt', "a")
for r in rewards:
    file.write(str(r) + ' ')
file.write('\n')
file.close()