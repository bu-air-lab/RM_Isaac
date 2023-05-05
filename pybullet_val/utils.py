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
    elif(gait_type == 'half_bound'):

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator

#Dont use type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

def get_foot_string(foot_index):

    if(foot_index == 0):
        return "FR"
    elif(foot_index == 1):
        return "FL"
    elif(foot_index == 2):
        return "BR"
    else:
        return "BL"


#Generate foot contact diagram
def gen_foot_contact_diagram(foot_contacts, gait, rm_transition_timesteps):

    start_frame = 0
    end_frame = len(foot_contacts)

    foot_contacts = foot_contacts[start_frame:end_frame]
    #rm_transition_timesteps = [x-start_frame for x in rm_transition_timesteps if x >= start_frame and x < end_frame]

    #Contains an entry for each foot contact period in format (start_contact, end_contact, foot_name)
    data = []

    #Keep track of which feet are making contact
    isContact = [False, False, False, False]

    #Keep track of timestep each foot made contact last
    start_contact_index = [-1, -1, -1, -1]

    #This is probably inneficient. Trajectories are small enough, so who cares.
    for t in range(len(foot_contacts)):

        for foot_index in range(4):

            #Foot just made contact
            if(foot_contacts[t][foot_index] and not isContact[foot_index]):
                isContact[foot_index] = True
                start_contact_index[foot_index] = t

            #Foot stopped making contact
            if(not foot_contacts[t][foot_index] and isContact[foot_index]):
                isContact[foot_index] = False
                data.append((start_contact_index[foot_index], t, get_foot_string(foot_index)))
                start_contact_index[foot_index] = -1


    for foot_index in range(4):
        if(isContact[foot_index]):
            data.append((start_contact_index[foot_index], len(foot_contacts), get_foot_string(foot_index)))

    #Now generate plot
    #cats = {"FR" : 1, "FL" : 1.5, "BR" : 2, "BL" : 2.5}
    cats = {"FR" : 0.1, "FL" : 0.2, "BR" : 0.3, "BL" : 0.4}
    colormapping = {"FR" : "C0", "FL" : "C1", "BR" : "C2", "BL" : "C3"}

    verts = []
    colors = []

    offset = 0.025
    for d in data:
       v =  [(d[0], cats[d[2]]-offset),
             (d[0], cats[d[2]]+offset),
             (d[1], cats[d[2]]+offset),
             (d[1], cats[d[2]]-offset)]
       verts.append(v)
       colors.append(colormapping[d[2]])

    bars = PolyCollection(verts, facecolors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(bars)
    ax.autoscale()

    #ax.set_yticks([1,1.5,2,2.5])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["FR", "FL", "BR", "BL"])

    #Denote RM transitions as vertical line for each timestep
    #print(rm_transition_timesteps)
    #for t in rm_transition_timesteps:
        #if(t == 10):
        #plt.axvline(x=t, color='k', linewidth=5, linestyle='--')
        #else:
        #plt.axvline(x=t, color='k')

    #plt.xticks([0, 1, 2], ['January', 'February', 'March'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(gait+'_foot_contacts.pdf')

    return