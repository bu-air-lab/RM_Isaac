import gym
from gym import spaces

import pybullet as p
import time
import pybullet_data as pd
import numpy as np
import torch

import utils

MAX_TORQUE = 35.5
JOINT_EPSILON = 0.02


#FL, FR, RL, RR
INIT_MOTOR_ANGLES = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])


class BulletEnv(gym.Env):

    def __init__(self, gait, isGUI=False):

        self.num_privileged_obs = None
        self.num_envs = 1
        self.num_actions = 12

        self.estimated_state = None

        self.isGUI = isGUI
        self.gait = gait

        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30

        self.urdf_path = "a1/a1.urdf"

        if(self.isGUI):
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        else:
            p.connect(p.DIRECT)


        p.setAdditionalSearchPath(pd.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path,[0,0,0.42])#, useFixedBase=1)
        p.setGravity(0,0,-9.8)

        self.time_step = 0.005

        p.setTimeStep(self.time_step)

        #Order: FR, FL, RR, RL
        self.foot_joint_indicies = [5, 10, 15, 20]

        #old ordering
        self.motor_ids = [1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19]

        #new ordering
        self.reordered_motor_ids = [6, 8, 9, 1, 3, 4, 16, 18, 19, 11, 13, 14]        
        
        for joint_index in self.foot_joint_indicies:
            p.enableJointForceTorqueSensor(self.robot, joint_index)
     
        #Just needed for resetting
        self.init_position, self.init_orientation = p.getBasePositionAndOrientation(self.robot)

        self._joint_angle_upper_limits = np.array([0.802851455917, 4.18879020479, -0.916297857297]*4)
        self._joint_angle_lower_limits = np.array([-0.802851455917, -1.0471975512, -2.69653369433]*4)

        #Initialize orientation variables
        self.current_joint_angles = INIT_MOTOR_ANGLES
        self.current_joint_velocities = [0 for i in range(12)]

        #Counts number of actions taken
        self.current_timestep = 0

        #number of environment actions taken until isDone=True
        self.max_timestep = 1000

        #Number of simulation steps we take per environment step
        self.action_repeat = 4 

        #self.torque_limits  = [100, 100, 100]*4
        self.torque_limits = [20, 55, 55]*4
        self.p_gains = 20
        self.d_gains = 0.5

        self.clip_action = 100

        self.current_RM_state = 1
        self.rm_transition_iters = 0


    def compute_torques(self, action):


        scaled_action = [0.25*x.item() for x in action]

        target_pos = list(scaled_action) + INIT_MOTOR_ANGLES - self.current_joint_angles

        P = [self.p_gains*x for x in target_pos] 
        D = [self.d_gains*x for x in self.current_joint_velocities]
        torques = [P[i] - D[i] for i in range(12)]
        return np.clip(torques, [-x for x in self.torque_limits], self.torque_limits)


    def _StepInternal(self, action):

        torques = self.compute_torques(action)

        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.reordered_motor_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=torques)

        p.stepSimulation()

        if(self.isGUI): #Sleep for rendering only
            time.sleep(self.time_step*2)

            base_pos = p.getBasePositionAndOrientation(self.robot)[0]
            camInfo = p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

        #Update joint angles and velocities
        joint_states = p.getJointStates(self.robot, self.reordered_motor_ids)
        joint_angles = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]

        self.current_joint_angles = joint_angles
        self.current_joint_velocities = joint_velocities

        try:
            self._ValidateMotorStates(torques)
        except Exception as e:
            print(e)
            exit()


    def step(self, action):

        isDone = False

        action = torch.clip(action, -self.clip_action, self.clip_action)

        #Take action_repeat number of simulation steps to complete action
        for i in range(self.action_repeat):
            self._StepInternal(action)


        self.current_timestep += 1

        self.state = self.getState(action)

        #Done if we reached the final timestep, or if we flip over
        if(self.current_timestep == self.max_timestep or not self.isHealthy()):
            isDone = True

        reward = 0
        info = {}

        return self.state, reward, isDone, info

    def reset(self):

        p.resetBasePositionAndOrientation(self.robot, self.init_position, self.init_orientation)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        #Re-Initialize joint positions
        self.resetJoints()

        #Let robot start falling
        for i in range(4):
            p.stepSimulation()


        #Reset camera
        if(self.isGUI):
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])


        #self.current_joint_angles = pos
        #self.current_joint_velocities = vel
        self.current_joint_angles = INIT_MOTOR_ANGLES
        self.current_joint_velocities = [0 for i in range(12)]

        self.current_timestep = 0

        #initial action of zeros
        #Call this to update joint positions and velocities
        self.state = self.getState(torch.zeros(12))

        #Actual starting state should be all zeros
        self.state = [0 for x in range(len(self.state))]

        self.rm_transition_iters = 0

        return self.state, None

    #Compute new RM state and update rm_transition_iters
    def _get_RM_state(self, max_rm_iters):

        #Find which feet are making contact with the ground
        contact_points = p.getContactPoints(self.robot, self.plane)

        foot_contact_vector = [0, 0, 0, 0]

        for point in contact_points:
          robot_link = point[3]
          for index,foot in enumerate(self.foot_joint_indicies):
            if(robot_link == foot):
                foot_contact_vector[index] = 1

        #Update rm_transition_iters before determing new RM state
        self.rm_transition_iters += 1

        #Compute new RM state based on current foot contacts, rm_transition_iters, and foot heights
        new_rm_state = utils.get_RM_state(self.current_RM_state, 
                                            foot_contact_vector, 
                                            self.rm_transition_iters, 
                                            self.gait, 
                                            max_rm_iters,
                                            self.estimated_state)

        #Reset rm_transition_iters when new RM state reached
        if(self.current_RM_state != new_rm_state):
            self.rm_transition_iters = 0

        self.current_RM_state = new_rm_state

        print("iter:", self.rm_transition_iters, foot_contact_vector, new_rm_state)

        #Determine one-hot RM state encoding based on gait type
        rm_state_encoding = []
        if(self.gait == 'trot' or self.gait == 'pace' or self.gait == 'bound'):

            if(self.current_RM_state == 1):
                rm_state_encoding.extend([1, 0])
            else:
                rm_state_encoding.extend([0, 1])

        elif(self.gait == 'walk' or self.gait == 'three_one'):

            if(self.current_RM_state == 1):
                rm_state_encoding.extend([1, 0, 0, 0])
            elif(self.current_RM_state == 2):
                rm_state_encoding.extend([0, 1, 0, 0])
            elif(self.current_RM_state == 3):
                rm_state_encoding.extend([0, 0, 1, 0])
            elif(self.current_RM_state == 4):
                rm_state_encoding.extend([0, 0, 0, 1])

        elif(self.gait == 'biped_bound'):

            if(self.current_RM_state == 1):
                rm_state_encoding.extend([1, 0, 0, 0, 0])
            elif(self.current_RM_state == 2):
                rm_state_encoding.extend([0, 1, 0, 0, 0])
            elif(self.current_RM_state == 3):
                rm_state_encoding.extend([0, 0, 1, 0, 0])
            elif(self.current_RM_state == 4):
                rm_state_encoding.extend([0, 0, 0, 1, 0])
            elif(self.current_RM_state == 5):
                rm_state_encoding.extend([0, 0, 0, 0, 1])

        else:
            print("UNSUPPORTED GAIT TYPE IN bullet_env.py")
            exit()

        return rm_state_encoding

    def getState(self, action):

        #Update current joint angles and velocities
        joint_states = p.getJointStates(self.robot, self.reordered_motor_ids)
        joint_angles = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]

        self.current_joint_angles = joint_angles
        self.current_joint_velocities = joint_velocities

        #Compute state
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)


        rm_iters = 6
        if(self.current_timestep >= 200 and self.current_timestep < 400):
            rm_iters = 8
        elif(self.current_timestep >= 400):
            rm_iters = 10
        # elif(self.current_timestep >= 600 and self.current_timestep < 800):
        #     rm_iters = 8
        # elif(self.current_timestep >= 800 and self.current_timestep < 1000):
        #     rm_iters = 9
        # elif(self.current_timestep >= 1000 and self.current_timestep < 1200):
        #     rm_iters = 10
        rm_state = self._get_RM_state(rm_iters)

        command = [1, 0] #forward
        #command = [1.0, 0] #forward
        #command = [1, 0.05] #forward + left
        #command = [1.5, -0.05] #forward + right

        #command = [-2, 0] #backward
        #command = [0, 0] #right
        #command = [2, -0.15] #turn left




        state = []
        state.extend(command) #Commands scale is (2, 2, 0.25). Command is [1, 0, 0]
        state.extend(self.current_joint_angles - INIT_MOTOR_ANGLES) #Joint angles offset
        state.extend([x*0.05 for x in self.current_joint_velocities])  #Joint velocities
        state.extend(action.tolist())
        state.extend(rm_state)
        state.append(self.rm_transition_iters * 0.1)
        state.append(rm_iters * 0.1)
        #Add dimensions for estimated base vel and foot heights
        state.extend([0, 0, 0, 0, 0, 0, 0])

        return state    


    def resetJoints(self, reset_time=1.0):


        for index,_id in enumerate(self.reordered_motor_ids):

            #Reset joint angles
            p.resetJointState(self.robot, _id, INIT_MOTOR_ANGLES[index], targetVelocity=0)

            #Disable default motor to allow torque control later on
            p.setJointMotorControl2(self.robot, _id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    #Must be balanced, only feet are allowed to touch ground, and body height must be above threshold
    def isHealthy(self):

        #Check balance
        isBalanced = True
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        euler_orientation = p.getEulerFromQuaternion(orientation)
        if(euler_orientation[0] > 0.4 or euler_orientation[1] > 0.2):
            isBalanced = False

        #Check if anything except feet are in contact with ground
        isContact = False
        contact_points = p.getContactPoints(self.robot, self.plane)
        links = []
        for point in contact_points:
            robot_link = point[3]
            if(robot_link not in self.foot_joint_indicies):
                isContact = True

        isHealthy = (isBalanced and not isContact)

        return isHealthy

 


    def _ValidateMotorStates(self, torques):
        # Check torque.
        if any(np.abs(torques) > MAX_TORQUE):
            raise Exception("Torque limits exceeded\ntorques: {}".format(torques))

        # Check joint positions.
        if (any(self.current_joint_angles > (self._joint_angle_upper_limits + JOINT_EPSILON)) or
            any(self.current_joint_angles < (self._joint_angle_lower_limits - JOINT_EPSILON))):
            #print("Joint angle limits exceeded\nangles: {}".format(self.current_joint_angles))
            raise Exception("Joint angle limits exceeded\nangles: {}".format(self.current_joint_angles))

    def setEstimatedState(self, state):
        self.estimated_state = state