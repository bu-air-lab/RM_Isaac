import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator

import utils

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

def get_max_RM_iters(gait):

	if(gait == 'trot' or gait == 'bound'):
		return 8
	elif(gait == 'pace'):
		return 10
	elif(gait == 'walk'):
		return 5

	#bound_walk
	return 6


#Given foot contact sequences and gait type, return time steps where rm transition occured
def get_rm_transitions(foot_contacts, gait):

	current_RM_state = 1
	rm_transition_iters = 1

	max_rm_iters = get_max_RM_iters(gait)

	rm_transitions = []

	for t in range(len(foot_contacts)):

		new_RM_state = utils.get_RM_state(current_RM_state, foot_contacts[t], rm_transition_iters, gait, max_rm_iters)

		if(new_RM_state == current_RM_state):
			rm_transition_iters += 1
		else:
			rm_transition_iters = 1
			rm_transitions.append(t)

		current_RM_state = new_RM_state

	return rm_transitions


gait = 'bound_walk'

foot_contacts = []

file = open(gait + '_foot_contacts.txt', 'r')
lines = file.readlines()

#Order: FR, FL, RR, RL
for line in lines:

	#List of strings
	foot_contact_lst = line.strip('[\n')[:-1].split()

	#Convert to list of bools
	for i,contact in enumerate(foot_contact_lst):
		if(contact == 'True'):
			foot_contact_lst[i] = True
		else:
			foot_contact_lst[i] = False

	foot_contacts.append(foot_contact_lst)



#Take some random period of contacts
#foot_contacts = foot_contacts[200:275]
foot_contacts = foot_contacts[:75]

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
			data.append((start_contact_index[foot_index], t-1, get_foot_string(foot_index)))
			start_contact_index[foot_index] = -1

	#Add final points to data when loop is done
	if(t == len(foot_contacts) - 1):

		for foot_index in range(4):

			if(isContact[foot_index]):
				data.append((start_contact_index[foot_index], t-1, get_foot_string(foot_index)))

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
rm_transitions = get_rm_transitions(foot_contacts, gait)
print("RM transitions:", rm_transitions)

for t in rm_transitions:
	plt.axvline(x=t, color='k')
	#if(t == 10):
		#plt.axvline(x=t, color='k', linewidth=5, linestyle='--')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel("Timestep")
plt.tight_layout()

if(gait == 'bound_walk'):
	gait = 'Three-One'

plt.title(gait.capitalize() + " Foot Contact Plot")
plt.savefig(gait + '_foot_contacts.pdf', bbox_inches='tight')
