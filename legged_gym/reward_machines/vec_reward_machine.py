"""
Vectorized RM.
Only 2 states, intentionally built for compatibility with Isaac Gym (maintain each envs RM state on GPU)
"""


from legged_gym.reward_machines.reward_functions import *

class VecRewardMachine:

    def __init__(self, num_envs, device, bonus=1000):
        self.num_envs = num_envs
        self.rm_rews = torch.zeros(self.num_envs, device=device, dtype=torch.float)
        self.bonus=bonus


    def get_next_states(self, current_states, true_props):

        next_states = torch.clone(current_states)

        #Update from q0 -> q1 if true_props = 1
        q0_q1_indicies = (true_props == 1).nonzero()
        next_states[q0_q1_indicies] = 1

        #Update from q1 -> q0 if true_props = 2
        q1_q0_indicies = (true_props == 2).nonzero()
        next_states[q1_q0_indicies] = 0

        return next_states

    def get_reward(self, current_states, next_states, s_info):

        #Populate rm_rews with all self-loop rewards
        self.rm_rews = s_info['computed_reward']

        #Find environments that had an RM transition, and replace rm_rews with bonus reward
        bonus_envs = (current_states - next_states).nonzero()
        self.rm_rews[bonus_envs] *= self.bonus

        return self.rm_rews

    def step(self, current_states, true_props, s_info):
        """
        Emulates a step on the reward machines from current_states when observing *true_props*.
        """

        # Computing the next RM state per env
        next_states = self.get_next_states(current_states, true_props)

        #Getting the reward
        rew = self.get_reward(current_states, next_states, s_info)

        return next_states, rew


