#Given current RM state, new foot contacts, and gait type, return new RM state
#Foot contacts order: FR, FL, RR, RL
#Foot height order: FL, FR, RL, RR
def get_RM_state(current_RM_state, foot_contacts, rm_transition_iters, gait_type, max_rm_iters, estimated_foot_heights):

    #Determine which feet are sufficiently in the air
    foot_air_tmp = [0, 0, 0, 0]
    if(estimated_foot_heights != None):
        estimated_foot_heights = estimated_foot_heights[0][-4:].tolist()
        for i in range(4):
            if(estimated_foot_heights[i] >= 0.03):
                foot_air_tmp[i] = 1

    #Reorder foot heights to FR, FL, RR, RL (same as foot contacts)
    foot_air = [0, 0, 0, 0]
    foot_air[0] = foot_air_tmp[1]
    foot_air[1] = foot_air_tmp[0]
    foot_air[2] = foot_air_tmp[3]
    foot_air[3] = foot_air_tmp[2]


    if(gait_type == 'trot'):

        if(current_RM_state == 1):

            if(foot_contacts[1] and foot_contacts[2]
                and foot_air[0] and foot_air[3]
                and not foot_contacts[0] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        else:

            if(foot_contacts[0] and foot_contacts[3]
                and foot_air[1] and foot_air[2]
                and not foot_contacts[1] and not foot_contacts[2]
                and rm_transition_iters >= max_rm_iters):
                return 1
            else:
                return 2


    elif(gait_type == 'bound'):

        if(current_RM_state == 1):

            if(foot_contacts[0] and foot_contacts[1]
                and foot_air[2] and foot_air[3]
                and not foot_contacts[2] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        else:

            if(foot_contacts[2] and foot_contacts[3]
                and foot_air[0] and foot_air[1]
                and not foot_contacts[0] and not foot_contacts[1]
                and rm_transition_iters >= max_rm_iters):
                return 1
            else:
                return 2

    elif(gait_type == 'pace'):

        if(current_RM_state == 1):

            if(foot_contacts[1] and foot_contacts[3]
                and foot_air[0] and foot_air[2]
                and not foot_contacts[0] and not foot_contacts[2]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        else:

            if(foot_contacts[0] and foot_contacts[2]
                and foot_air[1] and foot_air[3]
                and not foot_contacts[1] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 1
            else:
                return 2

    #FR, FL, RR, RL
    elif(gait_type == 'walk'):

        if(current_RM_state == 1):

            if(foot_contacts[0] and foot_contacts[2] and foot_contacts[3]
                and foot_air[1]
                and not foot_contacts[1]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        elif(current_RM_state == 2):

            if(foot_contacts[0] and foot_contacts[1] and foot_contacts[3]
                and foot_air[2]
                and not foot_contacts[2]
                and rm_transition_iters >= max_rm_iters):
                return 3
            else:
                return 2

        elif(current_RM_state == 3):

            if(foot_contacts[1] and foot_contacts[2] and foot_contacts[3]
                and foot_air[0]
                and not foot_contacts[0]
                and rm_transition_iters >= max_rm_iters):
                return 4
            else:
                return 3

        elif(current_RM_state == 4):

            if(foot_contacts[0] and foot_contacts[1] and foot_contacts[2]
                and foot_air[3]
                and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 1
            else:
                return 4


    elif(gait_type == 'three_one'):

        #FL/RR/RL -> FL -> FR/RL/RR -> FR -> ...
        if(current_RM_state == 1):

            if(foot_contacts[1] and foot_contacts[2] and foot_contacts[3]
                and foot_air[0]
                and not foot_contacts[0]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        elif(current_RM_state == 2):

            if(foot_contacts[1]
                and foot_air[0] and foot_air[2] and foot_air[3]
                and not foot_contacts[0] and not foot_contacts[2] and not foot_contacts[3]
                and rm_transition_iters >= int(max_rm_iters/2)):
                return 3
            else:
                return 2

        elif(current_RM_state == 3):

            if(foot_contacts[0] and foot_contacts[2] and foot_contacts[3]
                and foot_air[1]
                and not foot_contacts[1]
                and rm_transition_iters >= max_rm_iters):
                return 4
            else:
                return 3

        elif(current_RM_state == 4):

            if(foot_contacts[0]
                and foot_air[1] and foot_air[2] and foot_air[3]
                and not foot_contacts[1] and not foot_contacts[2] and not foot_contacts[3]
                and rm_transition_iters >= int(max_rm_iters/2)):
                return 1
            else:
                return 4

    #FL/FR -> RR -> FL/FR -> RL -> ...
    elif(gait_type == 'biped_bound'):

        if(current_RM_state == 1):

            if(foot_contacts[0] and foot_contacts[1]
                and foot_air[2] and foot_air[3]
                and not foot_contacts[2] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 2
            else:
                return 1

        elif(current_RM_state == 2):

            if(foot_contacts[2]
                and foot_air[0] and foot_air[1] and foot_air[3]
                and not foot_contacts[0] and not foot_contacts[1] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 3
            else:
                return 2

        elif(current_RM_state == 3):

            if(foot_contacts[0] and foot_contacts[1]
                and foot_air[2] and foot_air[3]
                and not foot_contacts[2] and not foot_contacts[3]
                and rm_transition_iters >= max_rm_iters):
                return 4
            else:
                return 3

        elif(current_RM_state == 4):

            if(foot_contacts[3]
                and foot_air[0] and foot_air[1] and foot_air[2]
                and not foot_contacts[0] and not foot_contacts[1] and not foot_contacts[2]
                and rm_transition_iters >= max_rm_iters):
                return 1
            else:
                return 4


    else:
        print("UNSUPPORTED GAIT TYPE IN utils.py")
        exit()