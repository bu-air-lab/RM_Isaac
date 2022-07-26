"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.

However, each environment must implement the following function:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.

Notes:
    - The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
    - The agent only gets the reward given by the RM.
    - Rewards coming from the environment are ignored.
"""

import gym
from gym import spaces
import numpy as np
from legged_gym.reward_machines.reward_machine import RewardMachine


#class RewardMachineEnv(gym.Wrapper):
class RewardMachineEnv():
    def __init__(self, env, rm_file):
        """
        RM environment
        --------------------
        It adds an RM to the environment:
            - This code keeps track of the current state on all RMs
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_file: string with path to the RM file.
        """
        #super().__init__(env)

        # Loading the reward machines
        self.rm_file = rm_file
        self.reward_machine = RewardMachine(rm_file)
        self.num_rm_states = len(self.reward_machine.get_states())

        # The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine

        ### TODO: consider env.obs_buffer instead? Re-run RM PyBullet code to see exactly what env.observation space is, and see what get_observation() returns
        #self.observation_dict  = spaces.Dict({'features': env.observation_space, 'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        #flatdim = gym.spaces.flatdim(self.observation_dict)
        #s_low  = float(env.observation_space.low[0])
        #s_high = float(env.observation_space.high[0])
        #self.observation_space = spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)

        # Computing one-hot encodings for the non-terminal RM states
        self.rm_state_features = {}
        for u_id in self.reward_machine.get_states():
            u_features = np.zeros(self.num_rm_states)
            u_features[len(self.rm_state_features)] = 1
            self.rm_state_features[u_id] = u_features
        self.rm_done_feat = np.zeros(self.num_rm_states) # for terminal RM states, we give as features an array of zeros

        print(self.rm_state_features)

    # Reseting some environments
    def reset_idx(self, env_ids):
        self.env.reset_idx(env_ids)
        self.current_u_id  = self.reward_machine.reset()

        # Adding the RM state to the observation
        self.get_observation(self.current_u_id, False)
        #return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False)

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        self.crm_params = self.obs, action, next_obs, env_done, true_props, info
        self.obs = next_obs

        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)

        # returning the result of this action
        done = rm_done or env_done
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, done)

        return rm_obs, rm_rew, done, info

    def get_observation(self, u_id, done):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[u_id]
        rm_obs = {'features': next_obs,'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)

    #Add RM state to observations
    def compute_observations(self):

        self.obs_buf = torch.cat((self.obs_buf, self.rm_state_features[u_id]), dim=-1)         

