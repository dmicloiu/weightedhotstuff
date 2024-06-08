import random
import numpy as np


## UTILS FOR REAL DATA OF NETWORK ENVIRONMENT COLLECTED FROM AWS CLOUDPING
# DATA GATHERED FROM CLOUDPING as of 2nd of June 2024 - 18:40 CEST

# 5 nodes -> Cape Town (af-south-1), Hong Kong (ap-east-1), Canada (ca-central-1),
# London (eu-west-2) and Northern California (us-west-1)
paper_networkTopology = [[6.46, 357.86, 222.56, 146.94, 285.3],
                   [361.26, 3.02, 197.7, 211.4, 156.33],
                   [228.36, 198.29, 3.94, 79.35, 80.43],
                   [152.98, 210.13, 78.42, 3.51, 147.77],
                   [290.1, 155.81, 79.21, 147.82, 4.17]]


# 8 nodes -> af-south-1, ap-east-1, ap-northeast-1, ap-south-1, ap-southeast-1, ca-central-1, eu-west-2, me-south-1
two_faults_networkTopology = [
    [6.46, 357.86, 352.94, 386.6, 360.55, 267.56, 323.36, 410.47],  # af-south-1
    [361.26, 3.02, 54.99, 39.24, 49.35, 96.15, 38.5, 130.9],        # ap-east-1
    [356.42, 54.63, 3.25, 35.12, 10.52, 134.51, 71.28, 111.76],     # ap-northeast-1
    [273.27, 93.86, 140.49, 3.75, 132.28, 128.04, 58.0, 150.3],     # ap-south-1
    [325.78, 38.07, 71.59, 74.19, 3.62, 56.32, 78.4, 94.9],         # ap-southeast-1
    [228.36, 198.29, 143.75, 174.86, 150.07, 3.94, 211.62, 152.98], # ca-central-1
    [152.98, 210.13, 210.27, 240.26, 216.13, 118.07, 3.51, 265.8],# eu-west-2
    [221.73, 128.19, 168.88, 158.65, 165.6, 40.44, 92.72, 2.13]     # me-south-1
]

# 11 nodes -> af-south-1, ap-east-1, ap-northeast-1, ap-south-1, ap-southeast-1, ca-central-1,
# eu-west-2, me-south-1, sa-east-1, us-east-1, us-west-1
three_faults_networkTopology = [[6.46, 357.86, 352.94, 386.6, 360.55, 267.56, 323.36, 410.47, 222.56, 153.93, 174.45],  # af-south-1
    [361.26, 3.02, 54.99, 39.24, 49.35, 96.15, 38.5, 130.9, 197.7, 218.1, 239.14],  # ap-east-1
    [356.42, 54.63, 3.25, 35.12, 10.52, 134.51, 71.28, 111.76, 143.96, 252.03, 274.73],  # ap-northeast-1
    [273.27, 93.86, 140.49, 128.04, 132.28, 3.75, 58.0, 150.3, 193.98, 128.38, 147.05],  # ap-south-1
    [325.78, 38.07, 71.59, 74.19, 78.4, 56.32, 3.62, 94.9, 211.39, 183.11, 203.4],  # ap-southeast-1
    [228.36, 198.29, 143.75, 174.86, 150.07, 197.43, 211.62, 199.99, 3.94, 93.79, 107.82],  # ca-central-1
    [152.98, 210.13, 210.27, 240.26, 216.13, 118.07, 172.46, 265.8, 78.42, 18.44, 33.34],  # eu-west-2
    [221.73, 128.19, 168.88, 158.65, 165.6, 40.44, 92.72, 186.88, 162.41, 92.08, 103.08],  # me-south-1
    [342.88, 309.61, 255.91, 286.87, 261.32, 304.2, 326.24, 311.63, 125.76, 203.89, 218.08],  # sa-east-1
    [230.22, 197.4, 157.76, 179.88, 152.2, 195.86, 228.36, 202.02, 16.92, 92.73, 112.57],  # us-east-1
    [290.1, 155.81, 107.95, 133.3, 109.38, 225.49, 170.04, 139.76, 79.21, 153.37, 171.54]]   # us-west-1


## UTILS FOR GENERATING THE LATENCY VECTORS
def generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation=[]):
    Lphases = []
    for viewNumber in range(numberOfViews):
        Lphases.append(generateLatenciesToLeader(n, leaderRotation[viewNumber], networkTopology))
    return Lphases

## UTILS FOR LEADER ROTATION
def generateLatenciesToLeader(n, leaderID, networkTopology):
    # latency induced by the distances between the leaderID replica and rest of replicas
    L = networkTopology[leaderID]

    # for each type of message we add a transmission delay
    newview_delay = random.uniform(0, 5)
    Lnew_view = [0] * n
    for i in range(n):
        if i != leaderID:
            Lnew_view[i] = L[i] + newview_delay

    prepare_delay = random.uniform(0, 2)
    Lprepare = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprepare[i] = L[i] + prepare_delay

    precommit_delay = random.uniform(0, 2)
    Lprecommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprecommit[i] = L[i] + precommit_delay

    commit_delay = random.uniform(0, 2)
    Lcommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lcommit[i] = L[i] + commit_delay

    return (Lnew_view, Lprepare, Lprecommit, Lcommit)


def getLeaderRotation(n, numberOfViews, type="round robin", currentLeaderRotation=None):
    fullRotation = [0] * n

    # "round robin" style
    for i in range(n):
        fullRotation[i] = i

    if type == "random":
        random.shuffle(fullRotation)

    elif type == "neighbouring" and currentLeaderRotation is not None:
        fullRotation = currentLeaderRotation[:min(n, len(currentLeaderRotation))]
        swapping = np.random.choice(fullRotation, 2, replace=False)

        # swap the following indices
        idx1 = swapping[0]
        idx2 = swapping[1]

        # actually swap them
        temp = fullRotation[idx1]
        fullRotation[idx1] = fullRotation[idx2]
        fullRotation[idx2] = temp

    leaderRotation = []

    for i in range(numberOfViews):
        leaderRotation.append(fullRotation[i % n])

    return leaderRotation


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