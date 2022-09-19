from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1BoundingCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 42
        rm_iters = 10
        #rm_iters_curriculum = False
        #rm_iters_range = [8, 14]
        #rm_iters_range = [8, 9]
        #rm_iters_range = [11,12]
        #rm_iters_range = [15, 16]

    class terrain( LeggedRobotCfg.terrain ):
        #mesh_type = 'plane'
        #measure_heights = False        selected = True # select a unique terrain type and pass all arguments

        curriculum = True

        #Default tertain curriculum
        #terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

        #Select only easiest 2 terrain types
        terrain_proportions = [0.5, 0.5, 0, 0, 0]
        selected = False
        #terrain_kwargs =  { 'type': 'random_uniform_terrain', 'min_height': -0.1, 'max_height': 0.1, 'step': 0.1, 'downsampled_scale': 0.5} # Dict of arguments for selected terrain

        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

    class sim( LeggedRobotCfg.sim ):
        dt = 0.005

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        curriculum = False

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        #penalize_contacts_on = ["thigh", "calf"]
        #terminate_after_contacts_on = ["base"]
        terminate_after_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27
        dof_acc_curriculum = False
        class scales( LeggedRobotCfg.rewards.scales ):
            #torques = -0.0005
            torques = -0.0002
            dof_pos_limits = -10.0
            base_height = -50.0
            #feet_contact_forces = -2
            #orientation = -1.0

class A1BoundingCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'bounding_a1'
        max_iterations = 1000 # number of policy updates
        load_run = 'rm_bound3' # folder directly containing model files
        checkpoint = 1000 # saved model iter
