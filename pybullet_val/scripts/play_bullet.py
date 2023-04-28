from bullet_env.bullet_env import BulletEnv
from bullet_env.blank_env import BlankEnv

from rm_ppo.runners import OnPolicyRunner

import numpy as np
import torch
    
#Load env:
gait = "walk"
env = BulletEnv(gait=gait, isGUI=True)

#Load Policy
train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 
                'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 
                'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                'init_member_classes': {}, 
                'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                'runner': {'algorithm_class_name': 'PPO', 'checkpoint': 1000, 'experiment_name': 'bounding_a1', 'load_run': 'gpu_pace2', 'max_iterations': 1000, 
                'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                'runner_class_name': 'OnPolicyRunner', 'seed': 1}

ppo_runner = OnPolicyRunner(BlankEnv(gait), train_cfg_dict)

ppo_runner.load("/home/david/Desktop/RM_Isaac/pybullet_val/saved_models/walk4.pt")

policy, state_estimator = ppo_runner.get_inference_policy()
obs,_ = env.reset()

for env_step in range(800):
 
    obs = torch.Tensor(obs)

    #Add estimated base vel and foot heights to observation
    estimated_state = state_estimator(obs.unsqueeze(0))

    #Pass estimated state to env
    env.setEstimatedState(estimated_state)

    obs = torch.cat((obs[:-7], estimated_state.squeeze(0)),dim=-1)

    action = policy(torch.Tensor(obs)).detach()#.squeeze(0)

    obs, rew, done, info = env.step(action.detach())


#trot0: [1.5, 0]
#trot2: [1.5, 0.04]
#trot4: [1.5, 0.01]

#pace1: [1.5, -0.065]
#pace3: [1.5, 0.01]
#pace4: [1.5, 0.075]

#bound1: [1.5, 0.025]
#bound2: [1.5, 0]
#bound3: [1.5, -0.03]
#bound4: [1.5, -0.03]

#half_bound0: [1.5, 0.04]
#half_bound1: [1.5, 0.06]

#three_one0: [1.5, 0.025]
#three_one1: [1.5, 0.02]
#three_one2: [1.5, 0]
#three_one3: [1.5, 0.025]
#three_one4: [1.5, -0.02]


#walk1: [1, -0.02]
#walk3: [1, -0.02]
#walk4: [1, -0.02]