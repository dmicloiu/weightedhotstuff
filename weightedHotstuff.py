import random
import heapq
import time
import math
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def formQuorumWeighted(LMessageReceived, weights, quorumWeight):
    heap = []
    for replicaIdx in range(n):
        heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    while weight < quorumWeight:
        (recivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Hotstuff all relicas have the same weight
        agreementTime = recivingTime

    return agreementTime


def predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit):
    quorumWeight = 2 * (f + delta) + 1  # quorum formation condition

    # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
    tPREPARE = formQuorumWeighted(Lnew_view, weights, quorumWeight)

    # after quorum formation -> leader sends PREPARE messages

    # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
    tPRECOMIT = formQuorumWeighted(Lprepare, weights, quorumWeight)

    # after quorum formation -> leader sends PRE-COMMIT messages

    # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
    tCOMMIT = formQuorumWeighted(Lprecommit, weights, quorumWeight)

    # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
    tDECIDE = formQuorumWeighted(Lcommit, weights, quorumWeight)

    ## total time of a Hotstuff view run
    return (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)


def formQuorumBasic(LMessageReceived, quorumWeight):
    # sort based on the messages that arrive fastest
    LMessageReceived = sorted(LMessageReceived)

    weight = 0
    replicaIdx = 0
    agreementTime = 0
    while weight < quorumWeight:
        agreementTime = LMessageReceived[replicaIdx]
        weight += 1  ## in Basic Hotstuff all relicas have the same weight
        replicaIdx += 1

    return agreementTime


def predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit):
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition -> majority for egalitarian quorum

    # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
    tPREPARE = formQuorumBasic(Lnew_view, quorumWeight)

    # after quorum formation -> leader sends PREPARE messages

    # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
    tPRECOMIT = formQuorumBasic(Lprepare, quorumWeight)

    # after quorum formation -> leader sends PRE-COMMIT messages

    # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
    tCOMMIT = formQuorumBasic(Lprecommit, quorumWeight)

    # after quorum formation -> leader sends COMMIT messages

    # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
    tDECIDE = formQuorumBasic(Lcommit, quorumWeight)

    ## total time of a Hotstuff view run
    return (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)


# def generateLatenciesToLeader(n, leaderID, low, high):
#     L = [0] * n
#
#     for i in range(n):
#         if i != leaderID:
#             L[i] = random.randint(low, high)
#
#     return L

def convert_bestweights_to_rmax_rmin(best_weights, vmax):
    replicas = [i for i in range(len(best_weights))]
    r_max = []
    r_min = []
    for rep_id in replicas:
        if best_weights[rep_id] == vmax:
            r_max.append(replicas[rep_id])
        else:
            r_min.append(replicas[rep_id])
    return r_max, r_min

def predictLatencySimmulatedAnnealing(n, f, curWeights, Lnew_view, Lprepare, Lprecommit, Lcommit):
    rounds = 10

    latency = 0
    for _ in range(rounds):
        latency += predictLatencyWeightedHotstuff(n, f, curWeights, Lnew_view, Lprepare, Lprecommit, Lcommit)

    return latency / rounds

def simulated_annealing(n, f, delta, Lnew_view, Lprepare, Lprecommit, Lcommit, suffix):
    # for assessing the simulated annealing process
    start = time.time()

    # # declare a seed for this process
    # random.seed(500)

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting weighting assignment
    curWeights = [1] * n
    for i in range(2 * f):
        curWeights[i] = 1 + delta / f

    # satrting with leader replica 0
    curLeader = 0

    # get a baseline
    curLat = predictLatencyWeightedHotstuff(n, f, curWeights, Lnew_view, Lprepare, Lprecommit, Lcommit)
    bestLat = curLat

    # variable to retain
    bestLeader = -1
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        newLeader = curLeader

        while True:
            replicaFrom = random.randint(0, n - 1)
            if curWeights[replicaFrom] == 1 + delta / f:
                break
        while True:
            replicaTo = random.randint(0, n - 1)
            if replicaTo != replicaFrom:
                break

        if replicaFrom == curLeader:
            newLeader = replicaTo

        newWeights = curWeights.copy()
        newWeights[replicaTo] = curWeights[replicaFrom]
        newWeights[replicaFrom] = curWeights[replicaTo]

        newLat = predictLatencyWeightedHotstuff(n, f, curWeights, Lnew_view, Lprepare, Lprecommit, Lcommit)

        if newLat < curLat:
            curLeader = newLeader
            curWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLat - curLat) / temp):
                jumps = jumps + 1
                curLeader = newLeader
                curWeights = newWeights

        if newLat < bestLat:
            bestLat = newLat
            bestLeader = newLeader
            bestWeights = newWeights

        temp = temp * (1 - theta)
        step += 1

    end = time.time()
    r_max, r_min = convert_bestweights_to_rmax_rmin(bestWeights, vmax)

    # print('--------------------------------')
    # print('--------' + suffix + ' Simulated annealing')
    # print('--------------------------------')
    # print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    # print('Final solution latency:', bestLat)
    # print('Best Configuration:  R_max: {}, weight: {} | R_min: {}, weight: {} with leader {}'.format(r_max, vmax, r_min,
    #                                                                                                  vmin, bestLeader))
    # print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    # print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))

    return bestLat



f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

leaderID = 0

vmax = 1 + delta / f  # 2f replicas
vmin = 1  # n - 2f replicas

weights = []
for i in range(2 * f):
    weights.append(vmax)
for i in range(n - 2 * f):
    weights.append(vmin)

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

def generateLatenciesToLeader(n, leaderID):
    L = networkTopology[leaderID]
    # for each type of message we add a transmission delay
    newview_delay = random.uniform(0,5)
    Lnew_view = [0] * n
    for i in range(n):
        if i != leaderID:
            Lnew_view[i] = L[i] + newview_delay

    prepare_delay = random.uniform(0,2)
    Lprepare = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprepare[i] = L[i] + prepare_delay

    precommit_delay = random.uniform(0,2)
    Lprecommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprecommit[i] = L[i] + precommit_delay

    commit_delay = random.uniform(0,2)
    Lcommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lcommit[i] = L[i] + commit_delay

    return (Lnew_view, Lprepare, Lprecommit, Lcommit)

def generateExperimentLatencies(n, numberOfViews, leaderRotation=[]):
    Lphases = []
    for viewNumber in range(numberOfViews):
        Lphases.append(generateLatenciesToLeader(n, leaderRotation[viewNumber]))
    return Lphases

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

def generateAllPossibleLeaderRotations(n, numberOfViews):

    # leaderRotations = []
    # for i in range(n):
    #     leaderRotations.append([i])
    #
    # final = []
    # for subset in leaderRotations:
    #     if len(subset) == numberOfViews:
    #         final.append(subset)
    #         continue
    #
    #     last = subset[-1]
    #     for next in range(last + 1, n):
    #         leaderRotations.append(subset + [next])
    # return final
    leaderRotations = []
    for i in range(n):
        leaderRotations.append([i])

    for i in range(1, numberOfViews):
        newLeaderRotations = []
        for leaderRotation in leaderRotations:

            for replicaID in range(n):
                if replicaID not in leaderRotation:
                    newLeaderRotations.append(leaderRotation + [replicaID])

        leaderRotations = newLeaderRotations

    return leaderRotations


### EXPERIMENT 1
# print("------------ EXPERIMENT 1 ------------")
# networkTopology = generateNetworkTopology(n, 0, 1000)
# leaderRotation = [0]
# numberOfViews = 1
# Lphases = generateExperimentLatencies(n, numberOfViews, leaderRotation)
# (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]
#
# basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
# weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
#
# print(f"Basic Hotstuff with leader {leaderID} completes a view in {basicLatency}.")
# print(f"Weighted Hotstuff with leader {leaderID} completes a view in {weightedLatency}.")
# simulated_annealing(n, f, delta, Lnew_view, Lprepare, Lprecommit, Lcommit, suffix='')
# print("\n")

networkTopology = [[9.74, 290.12, 222.89, 149.97, 284.75],
                    [286.24, 2.21, 202.91, 210.16, 156.58],
                    [226.65, 203.8, 6.01, 81.86, 80.49],
                    [153.52, 210.18, 79.96, 5.1, 148.79],
                    [288.76, 155.68, 79.87, 148.54, 3.97]]

numberOfViews = 4

possibleLeaderRotations = generateAllPossibleLeaderRotations(n, numberOfViews)
print(possibleLeaderRotations)
latencies = []
for leaderRotation in possibleLeaderRotations:
    Lphases = generateExperimentLatencies(n, numberOfViews, leaderRotation)

    latency = 0
    for viewNumber in range(numberOfViews):
        (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]
        latency += predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
    latencies.append(latency)

print(latencies)
# Encode each unique rotation as a categorical variable
rotation_strings = ['-'.join(map(str, rotation)) for rotation in possibleLeaderRotations]

# Create DataFrames
df_rotations = pd.DataFrame(rotation_strings, columns=["rotation"])
df_latencies = pd.DataFrame(latencies, columns=["latency"])

print(df_rotations)
print(df_latencies)

# Combine the DataFrames
df = pd.concat([df_rotations, df_latencies], axis=1)

# Encode the rotation strings to numerical values
encoder = LabelEncoder()
encoded_rotations = encoder.fit_transform(df['rotation'])

# Add encoded rotations to the DataFrame
df['encoded_rotation'] = encoded_rotations

# Sort DataFrame by encoded_rotation for better visualization
df_sorted = df.sort_values(by='encoded_rotation')

# Plot the scatter plot with color gradient
plt.figure(figsize=(14, 8))
scatter = plt.scatter(df_sorted['encoded_rotation'], df_sorted['latency'], c=df_sorted['latency'], cmap='coolwarm', edgecolor='k', s=100)
plt.colorbar(scatter, label='Latency')

# Add title and labels
# plt.title('Analysis of Leader Rotation impact on Latency in Hotstuff', fontsize=16)
plt.xlabel('Encoded Leader Rotation', fontsize=14)
plt.ylabel('Latency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()


# basicLatency = []
#
# weightedLatency = []
# weightedFallback = []
#
# bestLatency = []
# bestFallback = []
#
# continuousLatency = []
# continuousFallback = []
#
# bestLeaderLatency = []
#
# viewNumbers = []
# for i in range(1, 20):
#     viewNumbers.append(i * 5)
#
# for numberOfViews in viewNumbers:
#     leaderRotation = getLeaderRotation(n, numberOfViews)
#     Lphases = generateExperimentLatencies(n, numberOfViews, leaderRotation)
#
#     # run in BASIC MODE
#     latency = 0
#     for viewNumber in range(numberOfViews):
#         (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]
#         latency += predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     latency /= numberOfViews
#     basicLatency.append(latency)
#     print("basic: {}".format(latency))
#
#     # run in WEIGHTED MODE
#     latency = 0
#     for viewNumber in range(numberOfViews):
#         (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]
#         latency += predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     latency /= numberOfViews
#     weightedLatency.append(latency)
#     print("weighted: {}".format(latency))
#
#     # run in BEST MODE
#     latency = 0
#     for viewNumber in range(numberOfViews):
#         (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]
#         latency += simulated_annealing(n, f, delta, Lnew_view, Lprepare, Lprecommit, Lcommit, suffix=' ')
#     latency /= numberOfViews
#     print("best: {}".format(latency))
#     bestLatency.append(latency)
#
#
# # Plot the analysis on different types of behaviours of Hotstuff
# plt.figure(figsize=(10, 8))
# plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
#          label='No Weights')
# plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Randomly assigned AWARE Weights')
#
# plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best assigned AWARE Weights')
#
# plt.title('Analysis of Average Latency per View in Hotstuff', fontsize=16)
# plt.xlabel('Number of views', fontsize=14)
# plt.ylabel('Average Latency per View [ms]', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# ### EXPERIMENT 2
# print("------------ EXPERIMENT 2 ------------")
# simulations = 10000
#
# averageDifference = 0
# timesBasicIsFaster = 0
# timesEqualPerformance = 0
#
# differences = []
# for _ in range(simulations):
#     Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#
#     basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
#
#     if basicLatency < weightedLatency:
#         timesBasicIsFaster += 1
#     elif basicLatency == weightedLatency:
#         timesEqualPerformance += 1
#     else:
#         differences.append(basicLatency - weightedLatency)
#         averageDifference += basicLatency - weightedLatency
#
# # compute the average
# averageDifference /= simulations
#
# print(f"Basic Hotstuff is faster than the weighted version in {timesBasicIsFaster} view simulations.")
# print(f"The two algorithms have equal performance in {timesEqualPerformance} simulations.")
# print(f"Weighted Hotstuff is on average with {averageDifference} faster than the Basic version.")
# print("\n")
#
# # debug purposes
# # print(differences)
#
# # Plot histogram
# plt.figure(figsize=(8, 6))
# plt.hist(differences, bins=50, color='skyblue', edgecolor='black')
# plt.axvline(x=averageDifference, color='red', linestyle='--', label=f'Average Difference: {averageDifference:.2f}')
# plt.title('Difference in Latency Basic vs Weighted Hotstuff')
# plt.xlabel('Difference')
# plt.ylabel('Number of Simulations')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# ### EXPERIMENT 3
# print("------------ EXPERIMENT 3 ------------")
#
# numberOfViews = 10
# leaderRotation = []
# for _ in range(numberOfViews):
#     leaderRotation.append(random.randint(0, n - 1))
#
# print(f"The experiment is conducted using {numberOfViews} views and the following leader rotation scheme {leaderRotation}.")
#
# basicLatency = 0
# weightedLatency = 0
# for viewNumber in range(numberOfViews):
#     leaderID = leaderRotation[viewNumber]
#
#     weightsOfView = []
#     for i in range(2 * f):
#         weightsOfView.append(vmax)
#     for i in range(n - 2 * f):
#         weightsOfView.append(vmin)
#
#     # the leader has to have vmax
#     if weightsOfView[leaderID] == vmin:
#         weightsOfView[leaderID] = vmax
#         ### since first 2f replicas have weight vmax is guaranteed that replica 0 has max weight
#         weightsOfView[0] = vmin
#
#     # construct latency vectors retained by the leader
#     Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#
#     basicLatency += predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     weightedLatency += predictLatencyWeightedHotstuff(n, f, weightsOfView, Lnew_view, Lprepare, Lprecommit, Lcommit)
#
#
# print(f"Basic Hotstuff has on average {basicLatency / numberOfViews} latency over the simulated views.")
# print(f"Weighted Hotstuff has on average {weightedLatency / numberOfViews} latency over the simulated views.")
# print("\n")
#
#
# ### EXPERIMENT 4
# print("------------ EXPERIMENT 4 ------------")
#
# # generate the latency vectors - network setup for which we optimise the weighting assignment
# Lnew_view = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
# Lprepare = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
# Lprecommit = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
# Lcommit = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
#
# basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
# weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
#
# print(f"Basic Hotstuff yields latency of {basicLatency}.")
# print(f"Weighted Hotstuff yields latency of {weightedLatency}.")
# print("The performance of the simulated annealing weighted assignment for the given network setup:\n")
# simulated_annealing(n, f, delta, Lnew_view, Lprepare, Lprecommit, Lcommit, suffix='')
# print("\n")
#
# ### EXPERIMENT 5
# print("------------ EXPERIMENT 5 ------------")
#
# simulations = 10000
# averageBasicLatency = 0
# basicLatencies = []
# for _ in range(simulations):
#     Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#
#     latency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     averageBasicLatency += latency
#     basicLatencies.append(latency)
#
# averageBasicLatency /= simulations
#
# print(f"The average Basic Hotstuff Latency is {averageBasicLatency}.\n")
#
# # Plot histogram
# plt.figure(figsize=(8, 6))
# plt.hist(basicLatencies, bins=50, color='skyblue', edgecolor='black')
# plt.axvline(x=averageBasicLatency, color='red', linestyle='--', label=f'Average Basic Hotstuff Latency: {averageBasicLatency:.2f}')
# plt.title('Latency of Basic Hotstuff')
# plt.xlabel('Latency [ms]')
# plt.ylabel('Number of Simulations')
# plt.xlim([0, 4000])
# plt.ylim([0, 600])
# plt.legend()
# plt.grid(True)
# plt.show()
#
# ### EXPERIMENT 6
# print("------------ EXPERIMENT 6 ------------")
# simulations = 10000
# averageWeightedLatency = 0
# weightedLatencies = []
# for _ in range(simulations):
#     Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#     Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
#
#     latency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
#     averageWeightedLatency += latency
#     weightedLatencies.append(latency)
#
# averageWeightedLatency /= simulations
#
# print(f"The average Weighted Hotstuff Latency is {averageWeightedLatency}.")
#
# # Plot histogram
# plt.figure(figsize=(8, 6))
# plt.hist(weightedLatencies, bins=50, color='skyblue', edgecolor='black')
# plt.axvline(x=averageWeightedLatency, color='red', linestyle='--', label=f'Average Weighted Hotstuff Latency: {averageWeightedLatency:.2f}')
# plt.title('Latency of Weighted Hotstuff')
# plt.xlabel('Latency [ms]')
# plt.xlim([0, 4000])
# plt.ylim([0, 600])
# plt.ylabel('Number of Simulations')
# plt.legend()
# plt.grid(True)
# plt.show()