import numpy as np
import random as rm
import sys

states = ['Ice Cream', "McDonald's", 'Salad', 'Boba', 'Sushi', 'Beer']

def activity_forecast(days,states,input,transmat=None):
    # Choose the starting state
    if input not in states:
        sys.exit('input not valid')

    # Probabilities matrix (transition matrix)
    if transmat == None:
        print("No transition matrix given. Generating random matrix...")
        flag = False
        while flag == False:
            transmat = np.random.rand(len(states),len(states))
            transmat = transmat/transmat.sum(axis=1,keepdims=1)
            # even with this operation, due to float point operation,
            # sometimes not all rows add to 1. So keep generating until a good one comes out
            for i in range(len(states)):
                if sum(transmat[i]) == 1:
                    flag = True
                else:
                    flag = False
                    break

    elif np.shape(transmat) != (len(states),len(states)):
        sys.exit("provided transition matrix shape not correct.")

    for i in range(len(states)):
        if sum(transmat[i]) != 1:
            sys.exit(f"Probability doesn't sum to 1 on row {np.where(transmat== transmat[i])[0][0]+1} of provided transition matrix")
        else:
            pass

    print("transition matrix:")
    for i in transmat:
        print(f"{i}")

    current = input
    print("\n---------------------- Results ----------------------\n")
    print("Start state: " + current)
    # Shall store the sequence of states taken. So, this only has the starting state for now.
    statelist = [current]
    i = 0
    # To calculate the probability of the statelist
    prob = 1
    while i != days:
        workingindex = states.index(current)
        change = np.random.choice(transmat[workingindex],replace=True,p=transmat[workingindex])
        prob = prob*change
        newindex = np.where(transmat[workingindex] == change)[0][0]
        current = states[newindex]
        statelist.append(current)
        i += 1
    print("End state after "+ str(days) + " days: " + current)
    print("Transition sequence: " + str(statelist))
    print("Probability of the possible sequence of states: " + str(prob))


activity_forecast(10,states,'Salad')
