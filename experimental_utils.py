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
    [6.46, 357.86, 352.94, 267.56, 323.36, 222.56, 146.94, 214.46], # af-south-1
    [361.26, 3.02, 54.99, 96.15, 38.5, 197.7, 211.4, 125.83],       # ap-east-1
    [356.42, 54.63, 3.25, 134.51, 71.28, 134.51, 143.96, 166.24],   # ap-northeast-1
    [273.27, 93.86, 140.49, 3.75, 58.0, 193.98, 120.96, 39.89],     # ap-south-1
    [325.78, 38.07, 71.59, 56.32, 3.62, 211.39, 172.41, 92.18],     # ap-southeast-1
    [228.36, 198.29, 143.75, 197.43, 211.62, 3.94, 79.35, 163.95],  # ca-central-1
    [152.98, 210.13, 210.27, 118.07, 172.46, 78.42, 3.51, 89.37],   # eu-west-2
    [221.73, 128.19, 168.88, 40.44, 92.72, 162.41, 90.62, 2.13]     # me-south-1
]

# 11 nodes -> af-south-1, ap-east-1, ap-northeast-1, ap-south-1, ap-southeast-1, ca-central-1,
# eu-west-2, me-south-1, sa-east-1, us-east-1, us-west-1
three_faults_networkTopology = [
    [6.46, 357.86, 352.94, 267.56, 323.36, 222.56, 146.94, 214.46, 337.43, 226.84, 285.3],  # af-south-1
    [361.26, 3.02, 54.99, 96.15, 38.5, 197.7, 211.4, 125.83, 308.05, 200.24, 156.33],       # ap-east-1
    [356.42, 54.63, 3.25, 134.51, 71.28, 134.51, 143.96, 166.24, 257.28, 154.07, 108.77],   # ap-northeast-1
    [273.27, 93.86, 140.49, 3.75, 58.0, 193.98, 120.96, 39.89, 304.25, 195.51, 224.0],      # ap-south-1
    [325.78, 38.07, 71.59, 56.32, 3.62, 211.39, 172.41, 92.18, 324.63, 222.47, 170.53],     # ap-southeast-1
    [228.36, 198.29, 143.75, 197.43, 211.62, 3.94, 79.35, 163.95, 124.44, 17.57, 80.43],    # ca-central-1
    [152.98, 210.13, 210.27, 118.07, 172.46, 78.42, 3.51, 89.37, 185.66, 77.13, 147.77],    # eu-west-2
    [221.73, 128.19, 168.88, 40.44, 92.72, 162.41, 90.62, 2.13, 274.81, 167.56, 222.68],    # me-south-1
    [342.88, 309.61, 255.91, 304.2, 326.24, 125.76, 186.44, 275.94, 2.29, 115.7, 175.52],   # sa-east-1
    [230.22, 197.4, 157.76, 195.86, 228.36, 16.92, 79.7, 166.13, 114.95, 7.49, 64.46],      # us-east-1
    [290.1, 155.81, 107.95, 225.49, 170.04, 79.21, 147.82, 221.56, 175.25, 62.46, 4.17]     # us-west-1
]


## UTILS FOR GENERATING THE LATENCY VECTORS
def generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation=[]):
    Lphases = []
    for viewNumber in range(numberOfViews):
        Lphases.append(generateLatenciesToLeader(n, leaderRotation[viewNumber], networkTopology))
    return Lphases


## UTILS FOR WEIGHTED MODE AWARE WEIGHTS ASSIGNMENT
def set_up_weighting_scheme(networkTopology, delta, f):
    overall_distance_from_replicas = np.sum(networkTopology, axis=0)
    overall_distance_from_replicas = list(enumerate(overall_distance_from_replicas))

    # print(overall_distance_from_replicas)

    overall_distance_from_replicas = sorted(overall_distance_from_replicas, key=lambda x: x[1])

    n = 3 * f + 1 + delta
    weights = [1] * n
    idx = 0
    while idx < f:
        indexOfBestReplica = overall_distance_from_replicas[idx][0]
        weights[indexOfBestReplica] = 1 + delta / f

        indexOfWorstReplica = overall_distance_from_replicas[n - idx - 1][0]
        weights[indexOfWorstReplica] = 1 + delta / f

        idx += 1

    return weights

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

## UTILS FOR OTHERS
def calculate_improvement(basic, new):
    return ((new - basic) / basic) * 100