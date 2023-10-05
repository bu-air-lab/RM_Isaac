
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
env_cfg.env.num_envs = 1

env_cfg.terrain.mesh_type = 'plane'
env_cfg.terrain.measure_heights = False
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False

env_cfg.noise.add_noise = False

env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False

#Only vary commanded linear speed
env_cfg.commands.ranges.ang_vel_yaw = [0, 0]

args.experiment = 'rm'

speeds = [0.25, 0.5, 0.75, 1]


#Getting weird errors, running for each speed manually
speed = speeds[2]


#Set commanded speed
env_cfg.commands.ranges.lin_vel_x = [speed, speed]

#Prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

train_cfg.runner.resume = True
train_cfg.runner.checkpoint = 1000
train_cfg.runner.load_run = 'rm_'+args.gait+str(args.seed)
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = ppo_runner.get_inference_policy(device=env.device)

energy = 0
distance = 0

for i in range(int(env.max_episode_length)):

    actions = policy(obs.detach())
    obs, _, rews, dones, infos = env.step(actions.detach())

    torque = infos['torques']
    motor_vel = infos['dof_vels']
    energy_consumed = torch.sum(torch.abs(torque * motor_vel), dim=1).item()

    #Magnitude of velocity vector * 0.02 seconds per step
    distance_traveled = (torch.norm(infos['lin_vel']) * 0.02).item()

    energy += energy_consumed
    distance += distance_traveled

energy_per_meter = energy/distance

#Now append to file
file = open(args.gait + '_energy_per_meter.txt', "a")
file.write(str(energy_per_meter) + ' ')
file.write('\n')
file.close()
