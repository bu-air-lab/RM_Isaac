import math
import torch
import numpy as np

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

#Scale RM transition bonus based on direction
class PoseTransitionForward(RewardFunction):

    def __init__(self, bonus):
        super().__init__()
        self.bonus = bonus

    def get_type(self):
        return "pose_transition"

    #Only get a large bonus when we move far forward in x-direction
    def get_reward(self, s_info):

        x_bonus = (self.bonus * math.tanh(s_info['linear_velocity'][0])) #- 2*(self.bonus * abs(math.tanh(s_info['linear_velocity'][1])))
        return x_bonus 


#Reward A1 for moving forward, penalty for energy expenditure
class RewardA1Forward(RewardFunction):

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    #Balance reward for moving in x-direction while minimizing movement in y-direction and energy consumption
    def get_reward(self, s_info):

        forward_reward = s_info['linear_velocity'][0] #- 10*abs(s_info['linear_velocity'][1])
        #forward_reward = s_info['linear_velocity'][0]
        energy_penalty = 0.001* np.dot(s_info['joint_torques'], s_info['joint_velocities'])

        return forward_reward - energy_penalty


#Compute non-markovian environment reward
class NonMarkovianReward(RewardFunction):

    def __init__(self, bonus):
        super().__init__()
        self.bonus = bonus

    def get_type(self):
        return "nonmarkov_forward"

    #Balance reward for moving in x-direction while minimizing movement in y-direction and energy consumption
    #Only get a large bonus when we move far forward in x-direction
    #Reduced bonus for y-direction movement (positive or negative y-direction)
    def get_reward(self, s_info):

        reward = 0

        #achieved_poses = s_info['achieved_poses']
        #current_pose = s_info['current_pose']

        #Bonus reward for first time we hit P1, and first time we hit P2 (after P1)

        """if(achieved_poses == ['P1', 'P2']):
            reward = PoseTransitionForward(self.bonus).get_reward(s_info)
        else:
            reward = RewardA1Forward().get_reward(s_info)"""

        if(s_info['nonMarkovBonus']):
            reward = PoseTransitionForward(self.bonus).get_reward(s_info)
            #print("Bonus:", reward)
        else:
            reward = RewardA1Forward().get_reward(s_info)
            
        return reward


class Nogait(RewardFunction):

    def __init__(self, ignore):
        super().__init__()

    def get_type(self):
        return "nogait"

    def get_reward(self, s_info):
        return RewardA1Forward().get_reward(s_info)