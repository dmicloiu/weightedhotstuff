import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import time, math
from collections import deque
import experimental_utils
import plotly.graph_objects as plotly


# use the same function for Quorum Formation on both basic and weighted Chained Hotstuff
# we are using weights equal to 1 for the normal version
def formQuorumChained(LMessageReceived, weights, quorumWeight, faulty_replicas={}):
    heap = []
    for replicaIdx in range(n):
        # do not consider faulty replicas for consensus
        if replicaIdx in faulty_replicas:
            continue
        heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    while weight < quorumWeight:
        (recivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Chained Hotstuff all relicas have the same weight
        agreementTime = recivingTime

    return agreementTime


def setupChainedHotstuffSimulation(n, leaderRotation, type="basic", faulty=False):
    # set up the weighting scheme
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition

    if type != "basic":
        quorumWeight = 2 * (f + delta) + 1

    # basic means normal Chained Hotstuff -> all weights are 1
    # weighted means weighted Chained Hotstuff with the Vmax/Vmin AWARE weighting scheme

    # AWARE weighting scheme
    weights = [1] * n
    if type == "weighted" or type == "weighted-leader":
        weights = awareWeights
        # the rest of the weights are already Vmin = 1

    # consider the case we are making f replicas faulty
    faulty_replicas = set()
    if faulty:
        # find the highest weighted replicas and make them faulty -> MAXHEAP
        heap = []
        for replicaIdx in range(n):
            heapq.heappush(heap, (-weights[replicaIdx], replicaIdx))

        for _ in range(f):
            (weight, replicaIdx) = heapq.heappop(heap)
            faulty_replicas.add(replicaIdx)

    if type == "best" or type == "best-leader":
        return runChainedHotstuffSimulatedAnnealing(n, type, quorumWeight, leaderRotation, faulty_replicas)

    return runChainedHotstuffSimulation(n, type, weights, quorumWeight, leaderRotation, faulty_replicas)


def runChainedHotstuffSimulation(n, type, weights, quorumWeight, leaderRotation, faulty_replicas={}):
    # keep track of the total latency prediction of the simulation
    latency = 0

    # queue of commands that we are currently processing
    currentProcessingCommands = deque()
    for viewNumber in range(numberOfViews):
        # in each view we start a new proposal -> hence add a new command that is currently processed
        currentProcessingCommands.append(viewNumber)

        # given that the Hotstuff algorithm uses 5 phases which use 4 quorum formations, when we get to 5
        # currently processing commands, we need to pop wince the first command in queue is executed and hence finished
        if len(currentProcessingCommands) == 5:
            currentProcessingCommands.popleft()

        latency += processView(n, type, viewNumber, leaderRotation[viewNumber], currentProcessingCommands,
                               weights, quorumWeight, faulty_replicas)

    return latency


def processView(n, type, viewNumber, leaderID, currentProcessingCommands, weights, quorumWeight, faulty_replicas={}):
    # the total time it takes for performing the current phase for each command in the current view
    totalTime = 0

    for blockProposalNumber in currentProcessingCommands:
        # get the latency vector of the leader -> latency of leader receiving the vote message from each replica
        Lphase = Lphases[blockProposalNumber][viewNumber - blockProposalNumber]

        if type == "best-leader" or type == "weighted-leader" or type == "basic-leader":
            Lphase = generateLatenciesToLeader(n, leaderID)[viewNumber - blockProposalNumber]

        # EXECUTE the current phase of the command -> leader waits for quorum formation with (n - f) messages from replicas
        totalTime += formQuorumChained(Lphase, weights, quorumWeight, faulty_replicas)

    return totalTime


def runChainedHotstuffSimulatedAnnealing(n, type, quorumWeight, leaderRotation, faulty_replicas={}):
    # for assessing the simulated annealing process
    start = time.time()

    # # declare a seed for this process
    # random.seed(300)

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting weighting assignment
    currentWeights = [1] * n
    for i in range(2 * f):
        currentWeights[i] = vmax

    # get a baseline
    currentLatency = runChainedHotstuffSimulation(n, type, currentWeights, quorumWeight, leaderRotation, faulty_replicas)

    bestLatency = currentLatency
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        while True:
            replicaFrom = random.randint(0, n - 1)
            if currentWeights[replicaFrom] == vmax:
                break

        while True:
            replicaTo = random.randint(0, n - 1)
            if replicaTo != replicaFrom:
                break

        newWeights = currentWeights.copy()
        newWeights[replicaTo] = currentWeights[replicaFrom]
        newWeights[replicaFrom] = currentWeights[replicaTo]

        newLatency = runChainedHotstuffSimulation(n, type, currentWeights, quorumWeight, leaderRotation, faulty_replicas)

        if newLatency < currentLatency:
            currentWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeights = newWeights

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeights = newWeights

        temp = temp * (1 - theta)
        step += 1

    # DEBUG purposes
    # print(bestWeights)
    return bestLatency


def getLeaderRotation(n, numberOfViews, type="round robin", currentLeaderRotation=None):
    fullRotation = [0] * n

    # "round robin" style
    for i in range(n):
        fullRotation[i] = i

    if type == "random":
        random.shuffle(fullRotation)

    elif type == "neighbouring" and currentLeaderRotation is not None:
        swapping = random.choice(fullRotation, 2)

        # swap the following indices
        idx1 = swapping[0]
        idx2 = swapping[1]

        # actually swap them
        temp = currentLeaderRotation[idx1]
        currentLeaderRotation[idx1] = currentLeaderRotation[idx2]
        currentLeaderRotation[idx2] = temp

    leaderRotation = []

    for i in range(numberOfViews):
        leaderRotation.append(fullRotation[i % n])

    return leaderRotation


def chainedHotstuffOptimalLeader(n, numberOfViews, type='basic', faulty=False):
    # for assessing the simulated annealing process
    start = time.time()

    # # declare a seed for this process
    # random.seed(300)

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting leader rotation -> basic "round robin"
    currentLeaderRotation = getLeaderRotation(n, numberOfViews)

    # get a baseline
    currentLatency = setupChainedHotstuffSimulation(n, currentLeaderRotation, type + "-leader", faulty)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the leader rotation scheme
        # swap two leaders
        nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring")

        newLatency = setupChainedHotstuffSimulation(n, currentLeaderRotation, type + "-leader", faulty)

        if newLatency < currentLatency:
            currentLeaderRotation = nextLeaderRotation
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentLeaderRotation = nextLeaderRotation

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestLeaderRotation = nextLeaderRotation

        temp = temp * (1 - theta)
        step += 1

    # DEBUG purposes
    # print(besbestLeaderRotation)
    return bestLatency

def chainedHotstuffBestAndOptimalLeader(n, numberOfViews):
    # for assessing the simulated annealing process
    start = time.time()

    # # declare a seed for this process
    # random.seed(300)

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting weighting assignment
    currentWeights = [1] * n
    for i in range(2 * f):
        currentWeights[i] = vmax

    quorumWeight = quorumWeight = 2 * (f + delta) + 1

    # starting leader rotation -> basic "round robin"
    currentLeaderRotation = getLeaderRotation(n, numberOfViews)

    # get a baseline
    currentLatency =  runChainedHotstuffSimulation(n, "best-leader", currentWeights, quorumWeight, currentLeaderRotation)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation
    bestWeights = currentWeights

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # choose what we optimise in this step
        choice = 0  # optimise for leader
        if random.random() < 0.5:
            choice = 1  # optimise for weights

        if choice == 0:
            # generate "neighbouring" state for the leader rotation scheme
            # swap two leaders
            nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring")
            newLatency = runChainedHotstuffSimulation(n, "best-leader", currentWeights, quorumWeight, nextLeaderRotation)

            if newLatency < currentLatency:
                currentLeaderRotation = nextLeaderRotation
            else:
                rand = random.uniform(0, 1)
                if rand < math.exp(-(newLatency - currentLatency) / temp):
                    jumps = jumps + 1
                    currentLeaderRotation = nextLeaderRotation

            if newLatency < bestLatency:
                bestLatency = newLatency
                bestLeaderRotation = nextLeaderRotation


        else:
            while True:
                replicaFrom = random.randint(0, n - 1)
                if currentWeights[replicaFrom] == vmax:
                    break

            while True:
                replicaTo = random.randint(0, n - 1)
                if replicaTo != replicaFrom:
                    break

            newWeights = currentWeights.copy()
            newWeights[replicaTo] = currentWeights[replicaFrom]
            newWeights[replicaFrom] = currentWeights[replicaTo]

            newLatency = runChainedHotstuffSimulation(n, "best-leader", newWeights, quorumWeight, currentLeaderRotation)

            if newLatency < currentLatency:
                currentWeights = newWeights
            else:
                rand = random.uniform(0, 1)
                if rand < math.exp(-(newLatency - currentLatency) / temp):
                    jumps = jumps + 1
                    currentWeights = newWeights

            if newLatency < bestLatency:
                bestLatency = newLatency
                bestWeights = newWeights

        temp = temp * (1 - theta)
        step += 1

    # DEBUG purposes
    # print(besbestLeaderRotation)
    return bestLatency

def generateLatenciesToLeader(n, leaderID):
    # latency induced by the distances between the leaderID replica and rest of replicas
    L = networkTopology[leaderID]

    # for each type of message we add a transmission delay
    newview_delay = random.uniform(0, 5)
    Lnew_view = [0] * n
    for i in range(n):
        if i != leaderID:
            Lnew_view[i] = L[i] + newview_delay
        else:
            Lnew_view[i] = L[i]


    prepare_delay = random.uniform(0, 2)
    Lprepare = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprepare[i] = L[i] + prepare_delay
        else:
            Lprepare[i] = L[i]


    precommit_delay = random.uniform(0, 2)
    Lprecommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprecommit[i] = L[i] + precommit_delay
        else:
            Lprecommit[i] = L[i]

    commit_delay = random.uniform(0, 2)
    Lcommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lcommit[i] = L[i] + commit_delay
        else:
            Lcommit[i] = L[i]

    return [Lnew_view, Lprepare, Lprecommit, Lcommit]

def generateExperimentLatencies(n, numberOfViews, leaderRotation=[]):
    Lphases = []
    for viewNumber in range(numberOfViews):
        Lphases.append(generateLatenciesToLeader(n, leaderRotation[viewNumber]))
    return Lphases


f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

vmax = 1 + delta / f  # 2f replicas
vmin = 1  # n - 2f replicas
#
# numberOfViews = 10



# ## EXPERIMENT 1
# print("------------ EXPERIMENT 1 ------------")
# simulations = 10000
#
# avgBasicLatency = 0
# avgWeightedLatency = 0
#
# basicLatency = []
# weightedLatency = []
#
# for _ in range(simulations):
#     leaderRotation = getLeaderRotation(n, numberOfViews)
#
#     # run in BASIC MODE
#     latency = setupChainedHotstuffSimulation(n, leaderRotation)
#     basicLatency.append(latency)
#     avgBasicLatency += (latency / simulations)
#
#     # run in WEIGHTED MODE
#     latency = setupChainedHotstuffSimulation(n, leaderRotation, type="weighted")
#     weightedLatency.append(latency)
#     avgWeightedLatency += (latency / simulations)
#
#
# print(f"We perform Chained Hotstuff using {simulations} simulations of the protocol, using {numberOfViews} views.")
# print(f"Average latency of Basic Chained Hotstuff: {avgBasicLatency}")
# print(f"Average latency of Weighted Chained Hotstuff: {avgWeightedLatency}")
#
# xlim_left = min(min(basicLatency), min(weightedLatency))
# xlim_right = max(max(basicLatency), max(weightedLatency))
#
# # DEBUG purposes
# # print(xlim_left, xlim_right)
#
# # Graphical representation for BASIC MODE
# plt.figure(figsize=(8, 6))
# plt.hist(basicLatency, bins=50, color='skyblue', edgecolor='black')
# plt.axvline(x=avgBasicLatency, color='red', linestyle='--', label=f'Average Basic Latency: {avgBasicLatency:.2f}')
# plt.title('Latency of Chained Hotstuff')
# plt.xlabel('Latency [ms]')
# plt.ylabel('Number of Simulations')
# plt.xlim([xlim_left, xlim_right])
# plt.ylim([0, 700])
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Graphical representation for WEIGHTED MODE
# plt.figure(figsize=(8, 6))
# plt.hist(weightedLatency, bins=50, color='skyblue', edgecolor='black')
# plt.axvline(x=avgWeightedLatency, color='red', linestyle='--', label=f'Average Weighted Latency: {avgWeightedLatency:.2f}')
# plt.title('Latency of Weighted Chained Hotstuff')
# plt.xlabel('Latency [ms]')
# plt.ylabel('Number of Simulations')
# plt.xlim([xlim_left, xlim_right])
# plt.ylim([0, 700])
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
# basicLatency = []
# weightedLatency = []
# bestLatency = []
# bestLeaderLatency = []
#
# viewNumbers = []
# for i in range(1, 40):
#     viewNumbers.append(i * 5)
#
# for numberOfViews in viewNumbers:
#     leaderRotation = getLeaderRotation(n, numberOfViews)
#
#     # run in BASIC MODE
#     latency = setupChainedHotstuffSimulation(n, leaderRotation) / numberOfViews
#     basicLatency.append(latency)
#
#     # run in WEIGHTED MODE
#     latency = setupChainedHotstuffSimulation(n, leaderRotation, type="weighted") / numberOfViews
#     weightedLatency.append(latency)
#
#     # run in BEST MODE
#     latency = setupChainedHotstuffSimulation(n, leaderRotation, type="best") / numberOfViews
#     bestLatency.append(latency)
#
#     # run in WEIGHTED MODE with leader rotation optimisation
#     latency = chainedHotstuffOptimalLeader(n, numberOfViews, type="weighted") / numberOfViews
#     bestLeaderLatency.append(latency)
#
# # Plot the Analysis
# plt.figure(figsize=(10, 8))
# plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
#          label='No Weights')
# plt.plot(viewNumbers, weightedLatency, color='red', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Randomly assigned AWARE Weights')
# plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best assigned AWARE Weights')
# plt.plot(viewNumbers, bestLeaderLatency, color='orange', marker='*', linestyle='-.', linewidth=2, markersize=6,
#          label='Best leader rotation on Randomly assigned AWARE Weights')
#
# plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
# plt.xlabel('Number of views', fontsize=14)
# plt.ylabel('Average Latency per View [ms]', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# EXPERIMENT 3 -> faulty nodes
print("------------ EXPERIMENT 3 ------------")

# # 2nd of June 2024 - 18:40 CEST
# # 5 nodes -> af-south-1, ap-east-1, ca-central-1, eu-west-2, us-west-1
# networkTopology_1 = [[6.46, 357.86, 222.56, 146.94, 285.3],
#                    [361.26, 3.02, 197.7, 211.4, 156.33],
#                    [228.36, 198.29, 3.94, 79.35, 80.43],
#                    [152.98, 210.13, 78.42, 3.51, 147.77],
#                    [290.1, 155.81, 79.21, 147.82, 4.17]]
#
# # 11 nodes -> af-south-1, ap-east-1, ap-northeast-1, ap-south-1, ap-southeast-1, ca-central-1, eu-west-2, me-south-1, sa-east-1, us-east-1, us-west-1
# networkTopology_3 = [
#     [6.46, 357.86, 352.94, 386.6, 360.55, 267.56, 323.36, 410.47, 222.56, 153.93, 174.45, 147.31, 158.27, 146.94, 143.79, 214.46, 337.43, 226.84, 234.13, 285.3, 273.83],  # af-south-1
#     [361.26, 3.02, 54.99, 39.24, 49.35, 96.15, 38.5, 130.9, 197.7, 218.1, 239.14, 197.34, 252.85, 211.4, 202.36, 125.83, 308.05, 200.24, 187.08, 156.33, 147.08],  # ap-east-1
#     [356.42, 54.63, 3.25, 35.12, 10.52, 134.51, 71.28, 111.76, 143.96, 252.03, 274.73, 237.34, 200.75, 210.69, 216.71, 166.24, 257.28, 154.07, 133.84, 108.77, 97.42],  # ap-northeast-1
#     [273.27, 93.86, 140.49, 128.04, 132.28, 3.75, 58.0, 150.3, 193.98, 128.38, 147.05, 109.85, 129.74, 120.96, 113.05, 39.89, 304.25, 195.51, 200.34, 224.0, 216.34],  # ap-south-1
#     [325.78, 38.07, 71.59, 74.19, 78.4, 56.32, 3.62, 94.9, 211.39, 183.11, 203.4, 162.2, 239.13, 172.41, 161.91, 92.18, 324.63, 222.47, 200.35, 170.53, 160.11],  # ap-southeast-1
#     [228.36, 198.29, 143.75, 174.86, 150.07, 197.43, 211.62, 199.99, 3.94, 93.79, 107.82, 101.8, 69.78, 79.35, 86.56, 163.95, 124.44, 17.57, 25.44, 80.43, 60.41],  # ca-central-1
#     [152.98, 210.13, 210.27, 240.26, 216.13, 118.07, 172.46, 265.8, 78.42, 18.44, 33.34, 26.66, 12.71, 3.51, 10.61, 89.37, 185.66, 77.13, 88.39, 147.77, 130.08],  # eu-west-2
#     [221.73, 128.19, 168.88, 158.65, 165.6, 40.44, 92.72, 186.88, 162.41, 92.08, 103.08, 90.52, 106.33, 90.62, 87.04, 2.13, 274.81, 167.56, 171.45, 222.68, 213.87],  # me-south-1
#     [342.88, 309.61, 255.91, 286.87, 261.32, 304.2, 326.24, 311.63, 125.76, 203.89, 218.08, 212.91, 179.04, 186.44, 195.83, 275.94, 2.29, 115.7, 125.6, 175.52, 173.59],  # sa-east-1
#     [230.22, 197.4, 157.76, 179.88, 152.2, 195.86, 228.36, 202.02, 16.92, 92.73, 112.57, 102.77, 69.96, 79.7, 83.74, 166.13, 114.95, 7.49, 15.89, 64.46, 63.45],  # us-east-1
#     [290.1, 155.81, 107.95, 133.3, 109.38, 225.49, 170.04, 139.76, 79.21, 153.37, 171.54, 160.49, 130.86, 147.82, 144.32, 221.56, 175.25, 62.46, 52.75, 4.17, 22.61]   # us-west-1
# ]
#
# # 17 nodes -> af-south-1, ap-east-1, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-north-1, eu-south-1, eu-west-1, eu-west-2, me-south-1, sa-east-1, us-east-1, us-west-1
# networkTopology_5 = [
#     [6.46, 357.86, 352.94, 386.6, 360.55, 267.56, 323.36, 410.47, 222.56, 153.93, 174.45, 147.31, 158.27, 146.94, 143.79, 214.46, 337.43, 226.84, 234.13, 285.3, 273.83],  # af-south-1
#     [361.26, 3.02, 54.99, 39.24, 49.35, 96.15, 38.5, 130.9, 197.7, 218.1, 239.14, 197.34, 252.85, 211.4, 202.36, 125.83, 308.05, 200.24, 187.08, 156.33, 147.08],  # ap-east-1
#     [356.42, 54.63, 3.25, 35.12, 10.52, 134.51, 71.28, 111.76, 143.96, 252.03, 274.73, 237.34, 200.75, 210.69, 216.71, 166.24, 257.28, 154.07, 133.84, 108.77, 97.42],  # ap-northeast-1
#     [389.64, 38.23, 35.38, 4.51, 26.63, 131.19, 73.06, 145.67, 174.48, 256.52, 273.47, 234.77, 231.69, 240.81, 248.59, 159.69, 286.36, 179.37, 164.38, 133.42, 120.25],  # ap-northeast-2
#     [273.27, 93.86, 140.49, 128.04, 132.28, 3.75, 58.0, 150.3, 193.98, 128.38, 147.05, 109.85, 129.74, 120.96, 113.05, 39.89, 304.25, 195.51, 200.34, 224.0, 216.34],  # ap-south-1
#     [325.78, 38.07, 71.59, 74.19, 78.4, 56.32, 3.62, 94.9, 211.39, 183.11, 203.4, 162.2, 239.13, 172.41, 161.91, 92.18, 324.63, 222.47, 200.35, 170.53, 160.11],  # ap-southeast-1
#     [411.56, 130.01, 108.89, 146.71, 121.54, 154.78, 95.97, 4.4, 199.41, 276.25, 296.04, 256.57, 256.32, 266.11, 282.46, 186.56, 312.45, 201.33, 188.92, 139.55, 141.16],  # ap-southeast-2
#     [228.36, 198.29, 143.75, 174.86, 150.07, 197.43, 211.62, 199.99, 3.94, 93.79, 107.82, 101.8, 69.78, 79.35, 86.56, 163.95, 124.44, 17.57, 25.44, 80.43, 60.41],  # ca-central-1
#     [158.82, 218.84, 254.55, 255.93, 262.83, 128.55, 185.37, 277.4, 93.42, 4.55, 22.65, 14.22, 27.69, 19.22, 12.95, 91.51, 204.18, 93.68, 103.03, 154.92, 143.75],  # eu-central-1
#     [183.59, 241.84, 273.5, 274.66, 278.56, 148.85, 202.24, 295.38, 107.49, 23.37, 4.47, 30.72, 41.83, 34.1, 30.24, 103.28, 216.72, 112.67, 123.07, 171.06, 157.9],  # eu-north-1
#     [149.59, 197.27, 236.8, 233.37, 236.23, 109.01, 162.26, 254.29, 102.88, 12.59, 30.5, 2.29, 35.21, 26.84, 21.8, 88.49, 213.32, 103.75, 111.31, 160.82, 152.48],  # eu-south-1
#     [161.36, 254.11, 200.39, 232.07, 206.74, 128.25, 239.98, 256.72, 70.35, 27.47, 43.83, 37.58, 3.08, 13.88, 19.69, 103.25, 179.05, 69.84, 79.32, 130.02, 118.61],  # eu-west-1
#     [152.98, 210.13, 210.27, 240.26, 216.13, 118.07, 172.46, 265.8, 78.42, 18.44, 33.34, 26.66, 12.71, 3.51, 10.61, 89.37, 185.66, 77.13, 88.39, 147.77, 130.08],  # eu-west-2
#     [221.73, 128.19, 168.88, 158.65, 165.6, 40.44, 92.72, 186.88, 162.41, 92.08, 103.08, 90.52, 106.33, 90.62, 87.04, 2.13, 274.81, 167.56, 171.45, 222.68, 213.87],  # me-south-1
#     [342.88, 309.61, 255.91, 286.87, 261.32, 304.2, 326.24, 311.63, 125.76, 203.89, 218.08, 212.91, 179.04, 186.44, 195.83, 275.94, 2.29, 115.7, 125.6, 175.52, 173.59],  # sa-east-1
#     [230.22, 197.4, 157.76, 179.88, 152.2, 195.86, 228.36, 202.02, 16.92, 92.73, 112.57, 102.77, 69.96, 79.7, 83.74, 166.13, 114.95, 7.49, 15.89, 64.46, 63.45],  # us-east-1
#     [290.1, 155.81, 107.95, 133.3, 109.38, 225.49, 170.04, 139.76, 79.21, 153.37, 171.54, 160.49, 130.86, 147.82, 144.32, 221.56, 175.25, 62.46, 52.75, 4.17, 22.61]   # us-west-1
# ]
#
# networkTopologies = [networkTopology_1, networkTopology_3, networkTopology_5]
#
#
# # # EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
# basicLatency = []
# weightedLatency = []
# weightedLatencyFaulty = []
# bestLatency = []
# bestLatencyFaulty = []
# bestLeaderLatency = []
# weightedLatencyBestLeader = []
# bestLatencyBestLeader = []
#
# f_values = [1, 3]
# numberOfViews = 10
#
# for i, f in enumerate(f_values):
#     delta = 1  # additional replicas
#     n = 3 * f + 1 + delta  # total num of replicas
#
#     vmax = 1 + delta / f  # 2f replicas
#     vmin = 1  # n - 2f replicas
#
#     networkTopology = networkTopologies[i]
#
#     awareWeights = [1] * n
#     for i in range(2 * f):
#         awareWeights[i] = vmax
#
#     random.shuffle(awareWeights)
#     print(awareWeights)
#
#     leaderRotation = getLeaderRotation(n, numberOfViews)
#     Lphases = generateExperimentLatencies(n, numberOfViews, leaderRotation)
#
#     # # run in BASIC MODE
#     # latency = setupChainedHotstuffSimulation(n, leaderRotation) / numberOfViews
#     # basicLatency.append(latency)
#     #
#     # # run in WEIGHTED MODE
#     # latency = setupChainedHotstuffSimulation(n, leaderRotation, type="weighted") / numberOfViews
#     # weightedLatency.append(latency)
#     #
#     # # run in BEST MODE
#     # latency = setupChainedHotstuffSimulation(n, leaderRotation, type="best") / numberOfViews
#     # bestLatency.append(latency)
#     #
#     # # run in WEIGHTED MODE with leader rotation optimisation
#     # latency = chainedHotstuffOptimalLeader(n, numberOfViews, type="weighted") / numberOfViews
#     # bestLeaderLatency.append(latency)
#
#     basicLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='basic'))
#     weightedLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=False))
#     weightedLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=True)
#     bestLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='best'))
#     bestLatencyFaulty.append(setupChainedHotstuffSimulation(n, leaderRotation, type='best', faulty = True))
#     weightedLatencyBestLeader.append(chainedHotstuffOptimalLeader(n, numberOfViews, type='weighted'))
#     bestLatencyBestLeader.append(chainedHotstuffOptimalLeader(n, numberOfViews, type="best"))
#
#
#     # weightedLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=True)
#     # bestLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='best', faulty = True)
#     # bestLatencyBestLeader = chainedHotstuffOptimalLeader(n, numberOfViews, type="best")
#
# # Plot the Analysis
# plt.figure(figsize=(10, 8))
# plt.plot(f_values, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
#          label='No Weights')
# plt.plot(f_values, weightedLatency, color='red', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Non-faulty')
# plt.plot(f_values, weightedLatencyFaulty, color='darkred', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Faulty')
# plt.plot(f_values, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best Non-faulty')
# plt.plot(f_values, bestLatencyFaulty, color='darkgreen', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best Faulty')
# plt.plot(f_values, weightedLatencyBestLeader, color='orange', marker='*', linestyle='-.', linewidth=2, markersize=6,
#          label='Optimal Leader')
# plt.plot(f_values, bestLatencyBestLeader, color='darkorange', marker='*', linestyle='-.', linewidth=2, markersize=6,
#          label='Optimal Leader + Best')
#
# # plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
# plt.xlabel('# faults', fontsize=14)
# plt.ylabel('Average Latency per View [ms]', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
#

networkTopology = [[6.46, 357.86, 222.56, 146.94, 285.3],
                   [361.26, 3.02, 197.7, 211.4, 156.33],
                   [228.36, 198.29, 3.94, 79.35, 80.43],
                   [152.98, 210.13, 78.42, 3.51, 147.77],
                   [290.1, 155.81, 79.21, 147.82, 4.17]]

awareWeights = [2,1,2,1,1]
basicLatency = []
basicLatencyFaulty = []
weightedLatency = []
weightedLatencyFaulty = []
bestLatency = []
bestLatencyFaulty = []
bestLeaderLatency = []
weightedLatencyBestLeader = []
weightedLatencyBestLeaderFaulty = []
bestLatencyBestLeader = []
viewNumbers = []
for i in range(5, 21):
    viewNumbers.append(i)

for numberOfViews in viewNumbers:
    leaderRotation = getLeaderRotation(n, numberOfViews)
    Lphases = generateExperimentLatencies(n, numberOfViews, leaderRotation)

    print(numberOfViews)
    basicLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='basic') / numberOfViews)
    # basicLatencyFaulty.append(setupChainedHotstuffSimulation(n, leaderRotation, type='basic', faulty=True) / numberOfViews)

    weightedLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='weighted') / numberOfViews)
    weightedLatencyFaulty.append(setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=True) / numberOfViews)

    bestLatency.append(setupChainedHotstuffSimulation(n, leaderRotation, type='best') / numberOfViews)
    bestLatencyFaulty.append(setupChainedHotstuffSimulation(n, leaderRotation, type='best', faulty = True) / numberOfViews)

    weightedLatencyBestLeader.append(chainedHotstuffOptimalLeader(n, numberOfViews, type='weighted') / numberOfViews)
    weightedLatencyBestLeaderFaulty.append(chainedHotstuffOptimalLeader(n, numberOfViews, type='weighted', faulty=True) / numberOfViews)

    bestLatencyBestLeader.append(chainedHotstuffBestAndOptimalLeader(n, numberOfViews) / numberOfViews)

# Plot the Analysis
# plt.figure(figsize=(10, 8))
# plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
#          label='No Weights')
# plt.plot(viewNumbers, weightedLatency, color='red', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Randomly assigned AWARE Weights')
# plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best assigned AWARE Weights')
# # plt.plot(viewNumbers, bestLeaderLatency, color='orange', marker='*', linestyle='-.', linewidth=2, markersize=6,
# #          label='Best leader rotation on Randomly assigned AWARE Weights')

plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
         label='Basic')
plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Weighted')
# plt.plot(viewNumbers, weightedLatencyFaulty, color='darkred', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Faulty')
plt.plot(viewNumbers, weightedLatencyBestLeader, color='blue', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Optimal Leader Weighted')
plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best Weighted')
# plt.plot(viewNumbers, bestLatencyFaulty, color='black', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best Faulty')
plt.plot(viewNumbers, bestLatencyBestLeader, color='magenta', marker='D', linestyle='--', linewidth=2, markersize=6,
         label='(Optimal Leader + Best) Weighted')

# plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('#views', fontsize=12)
plt.ylabel('Average Latency per View [ms]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, weightedLatencyFaulty, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Weighted')

plt.plot(viewNumbers, weightedLatencyBestLeaderFaulty, color='blue', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Optimal Leader Weighted')

plt.plot(viewNumbers, bestLatencyFaulty, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best Weighted')

# plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('#views', fontsize=12)
plt.ylabel('Average Latency per View [ms]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate mean values
mean_basicLatency = np.mean(basicLatency)
mean_weightedLatency = np.mean(weightedLatency)
mean_bestLeaderLatency = np.mean(weightedLatencyBestLeader)
mean_bestLatency = np.mean(bestLatency)
mean_bestWeightsAndLeaderLatency = np.mean(bestLatencyBestLeader)

# Calculate percentage improvements
def calculate_improvement(basic, new):
    return ((new - basic) / basic) * 100

improvements = {
    'Weighted (Non-faulty)': calculate_improvement(mean_basicLatency, mean_weightedLatency),
    'Optimal Leader Weighted': calculate_improvement(mean_basicLatency, mean_bestLeaderLatency),
    'Best': calculate_improvement(mean_basicLatency, mean_bestLatency),
    'Optimal Leader Best': calculate_improvement(mean_basicLatency, mean_bestWeightsAndLeaderLatency),
}

# Generate LaTeX table code
latex_table = r"""
\begin{table}[h]
    \centering
    \caption{Percentage Improvement in Latency Compared to Basic Latency}
    \label{tab:latency_improvement}
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        & \textbf{Weighted (Non-faulty)} & \textbf{Optimal Leader Weighted} & \textbf{Best} & \textbf{Continuous} & \textbf{Optimal Leader Best} \\
        \hline
"""

# Assuming each row corresponds to different figures like in your provided example
for i, (key, value) in enumerate(improvements.items(), start=1):
    latex_table += f"        \\textbf{{Fig. {i}}} & {value:.2f}\\% \\\\ \n"
    latex_table += "        \hline\n"

latex_table += r"""
    \end{tabular}
\end{table}
"""

print(latex_table)

print(np.mean(basicLatency))
print(np.mean(weightedLatency))
print(np.mean(weightedLatencyBestLeader))
print(np.mean(bestLatency))
print(np.mean(bestLatencyBestLeader))

# basicLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='basic')
# weightedLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=False)
# weightedLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=True)
# bestLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='best')
# bestLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='best', faulty = True)
# weightedLatencyBestLeader = chainedHotstuffOptimalLeader(n, numberOfViews, type='weighted')
# bestLatencyBestLeader = chainedHotstuffOptimalLeader(n, numberOfViews, type="best")
#
# print(f"Simulation of Chained Hotstuff over {numberOfViews} views")
# print("---------------")
# print(f"No weights {basicLatency}")
# print("---------------")
# print(f"AWARE Weighted Scheme {weightedLatency}")
# print(f"AWARE Weighted Scheme FAULTY {weightedLatencyFaulty}")
# print("---------------")
# print(f"Simulated Annealing (Best) Weighting Scheme {bestLatency}")
# print(f"Simulated Annealing (Best) Weighting Scheme FAULTY {bestLatencyFaulty}")
# print("---------------")
# print(f"AWARE Weighted Scheme - BEST leader rotation {weightedLatencyBestLeader}")
# print(f"Simulated Annealing (Best) Weighting Scheme - BEST leader rotation {bestLatencyBestLeader}")
# print("---------------")
#
# results = {
#     "Non-faulty": round(weightedLatency, 2),
#     "Faulty": round(weightedLatencyFaulty, 2),
#     "Best": round(bestLatency, 2),
#     "Best Faulty": round(bestLatencyFaulty, 2),
#     "Optimal Leader": round(weightedLatencyBestLeader, 2),
#     "Optimal Leader Best": round(bestLatencyBestLeader, 2),
# }
#
# simulation_types = list(results.keys())
# simulation_results = list(results.values())
#
# plt.figure(figsize=(12, 10))
# bars = plt.barh(simulation_types, simulation_results, color=['skyblue', 'red', 'green', 'red', 'skyblue', 'green'])
# plt.xlabel('Latency [ms]', fontsize=14)
# # plt.ylabel('Weighting Assignment', fontsize=14)
# # plt.title(f'Weighted Chained Hotstuff performance over {numberOfViews} views', fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# # Adding values beside the bars
# for i, result in enumerate(simulation_results):
#     plt.text(result + 0.1, i, str(result), va='center', fontsize=10, style="oblique")
#
# plt.axvline(basicLatency, color='darkred', linestyle='--', linewidth=2, label=f'Chained Hotstuff (Baseline) = {basicLatency:.2f} ms')
#
# plt.gca().invert_yaxis()  # Invert y-axis to display simulation types from top to bottom
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.tight_layout()
# plt.legend(loc='upper right', fontsize=12)
# plt.show()
