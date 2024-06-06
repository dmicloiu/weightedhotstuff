import random
from datetime import timedelta
from typing import List

import requests
import random
from typing import List

import requests
import random


## UTILS FOR REAL DATA OF NETWORK ENVIRONMENT COLLECTED FROM AWS CLOUDPING



## UTILS FOR MOCKED DATA OF NETWORK ENVIRONMENT
def generateNetworkTopology(n, low, high):
    network = []

    for i in range(n):
        distanceReplicaI = []
        for j in range(n):
            if i == j:
                distanceReplicaI.append(0)
            elif i < j:
                distanceReplicaI.append(random.randint(low, high))
            else:
                distanceReplicaI.append(network[j][i])

        network.append(distanceReplicaI)
        # print(distanceReplicaI)

    return network

def tweakNetworkTopology(n, networkTopology):
    for i in range(n):
        offset = random.uniform(0, 20)
        for j in range(n):
            if i < j:
                networkTopology[i][j] = networkTopology[i][j] + random.uniform(-10, 10)
            else:
                networkTopology[i][j] = networkTopology[j][i]

    return networkTopology