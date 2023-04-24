import gym

#We need to pass an environment into OnPolicyRunner to load our policy for some reason
class BlankEnv(gym.Env):

    def __init__(self, gait):

        self.num_privileged_obs = None
        self.num_envs = 1
        self.num_actions = 12
        self.num_obs = 47

        if(gait == 'trot' or gait == 'pace' or gait == 'bound'):
            self.num_obs += 2
        elif(gait == 'walk' or gait == 'three_one'):
            self.num_obs += 4
        elif(gait == 'biped_bound'):
            self.num_obs += 5
        else:
            print("MUST IMPLEMENT NEW GAIT IN blank_env.py")
            exit()

    def reset(self):

        return None, None



