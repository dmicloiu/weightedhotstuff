import time, random, math
import numpy as np
import heapq
import matplotlib.pyplot as plt


def all_subsets(n):
    current_subsets = []

    for i in range(n):
        current_subsets.append([i])

    for subset in current_subsets:
        last_added = subset[-1]
        for i in range(last_added + 1, n):
            current_subsets.append(subset + [i])

    # DEBUG purposes
    # print(current_subsets)
    # print("--------------")

    return current_subsets


# function for computing the quorum weight such as the conditions for quorums are satisfied
def compute_quorum_weight(n, f, delta, weights):
    heap = []
    for replicaIdx in range(n):
        # push -weight in heap to make it a MAX HEAP since we are interested in the replicas holding most of the power
        heapq.heappush(heap, (-weights[replicaIdx], replicaIdx))

    # get out the f maximum weights -> worst case scenario all most powerful replicas fail
    for _ in range(f):
        heapq.heappop(heap)

    # when f most powerful replicas fail the others still need to form quorum -> AVAILABILITY
    quorum_weight = 0
    while len(heap) > 0:
        (weight, replicaIdx) = heapq.heappop(heap)
        quorum_weight -= weight

    # get all possible subsets which gather votes such that quorum is formed
    # check that they overlap by at least (f + 1) nodes -> CONSISTENCY
    possible_quorums = all_subsets(n)

    # out of the possible_quorums now select all quorums that have weight higher or equal than the quorum_weight
    valid_quorums = []
    for subset in possible_quorums:
        acquired_weight = 0

        for replicaIdx in subset:
            acquired_weight += weights[replicaIdx]

        if acquired_weight >= quorum_weight:
            valid_quorums.append(subset)

    for i in range(len(valid_quorums)):
        quorum1 = valid_quorums[i]
        for j in range(i + 1, len(valid_quorums)):
            quorum2 = valid_quorums[j]

            overlap = 0
            for element in quorum1:
                if element in quorum2:
                    overlap += 1

            if overlap < f + 1:
                # it means that the current weighting scheme is violating quorum system rules
                return -1

    # print(valid_quorums)
    return quorum_weight


def formQuorumWeighted(LMessageReceived, weights, quorumWeight, faulty_replicas={}):
    heap = []
    for replicaIdx in range(n):
        if replicaIdx not in faulty_replicas:
            heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    while weight < quorumWeight:
        # special case since we work with double numbers
        if len(heap) == 0 and abs(quorumWeight - weight) <= 1e-6:
            # prints for DEBUG purposes
            # print(quorumWeight)
            # print(weight)
            break

        (receivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Hotstuff all relicas have the same weight
        agreementTime = receivingTime

    return agreementTime


def predictLatency(weights, quorumWeight, leaderRotation=[], faulty_replicas={}, numberOfViews=1):
    latency = 0
    # predict latency for Hotstuff algorithm run with numberOfViews rounds
    for viewNumber in range(numberOfViews):
        (Lnew_view, Lprepare, Lprecommit, Lcommit) = generateLatenciesToLeader(n, leaderID=leaderRotation[viewNumber], low=0, high=1000)

        # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
        tPREPARE = formQuorumWeighted(Lnew_view, weights, quorumWeight, faulty_replicas)

        # after quorum formation -> leader sends PREPARE messages

        # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
        tPRECOMIT = formQuorumWeighted(Lprepare, weights, quorumWeight, faulty_replicas)

        # after quorum formation -> leader sends PRE-COMMIT messages

        # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
        tCOMMIT = formQuorumWeighted(Lprecommit, weights, quorumWeight, faulty_replicas)

        # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
        tDECIDE = formQuorumWeighted(Lcommit, weights, quorumWeight, faulty_replicas)

        latency += (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)


    # print(quorumWeight)
    # print(weights)
    # print(faulty_replicas)
    # print((tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE))
    # print("----------------------")

    ## total time of a Hotstuff run with numberOfViews rounds
    return latency

# there are 4 behaviours implemented for Hotstuff
# basic -> all weights are 1, hence it emulates the classic hotstuff behaviour
# weighted -> AWARE weighting scheme Hotstuff - first 2f are Vmax the rest are Vmin
# best -> AWARE weighting scheme Hotstuff - weights distributed with Simulated Annealing to minimise latency
# continuous -> continuous weighting scheme Hotstuff - Simulated Annealing approach to finding the best weighting scheme
def runWeightedHotstuff(n, f, delta, weightingScheme, leaderRotation=[], type="basic", faulty=False, numberOfViews=1):
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition -> majority for basic weighting scheme

    if type == "weighted" or type == "best":
        quorumWeight = 2 * (f + delta) + 1  # quorum formation condition

    elif type == "continunous":
        quorumWeight = compute_quorum_weight(n, f, delta, weightingScheme)

        if quorumWeight == -1:
            return 1e6

    faulty_replicas = set()
    if faulty:
        # find the highest weighted replicas and make them faulty -> MAXHEAP
        heap = []
        for replicaIdx in range(n):
            heapq.heappush(heap, (-weightingScheme[replicaIdx], replicaIdx))

        for _ in range(f):
            (weight, replicaIdx) = heapq.heappop(heap)
            faulty_replicas.add(replicaIdx)


    return predictLatency(weightingScheme, quorumWeight, leaderRotation=leaderRotation, faulty_replicas=faulty_replicas, numberOfViews=numberOfViews)

def continuousWeightedHotstuff(n, f, delta, leaderRotation=[], numberOfViews=1):
    # start the timer to register the convergence time
    start = time.time()

    # simulated annealing parameters
    step = 0
    step_max = 10000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2
    jumps = 0

    perturbation_step = 0.1

    # start with egalitarian quorum -> compute initial state
    currentWeightingScheme = [1] * n
    currentLatency = runWeightedHotstuff(n, f, delta, currentWeightingScheme, leaderRotation,
                                         type="continunous", numberOfViews=numberOfViews)
    currentLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, currentWeightingScheme, leaderRotation,
                                                   type="continunous", faulty=True, numberOfViews=numberOfViews)

    bestLatency = currentLatency
    bestLatencyWhenFaulty = currentLatencyWhenFaulty
    bestWeightingScheme = []

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the weighting scheme
        nextWeightingScheme = []
        for i in range(n):
            lowerBound = max(currentWeightingScheme[i] - perturbation_step, 0)
            upperBound = min(currentWeightingScheme[i] + perturbation_step, 2)
            nextWeightingScheme.append(random.uniform(lowerBound, upperBound))

        # compute the energy of the new state
        # -> in this case the latency of Hotstuff given the new weighted distribution scheme
        newLatency = runWeightedHotstuff(n, f, delta, nextWeightingScheme, leaderRotation,
                                         type="continunous", numberOfViews=numberOfViews)

        # if it performs better
        if newLatency < currentLatency:
            currentWeightingScheme = nextWeightingScheme
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeightingScheme = nextWeightingScheme

        if newLatency == bestLatency:
            newLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, nextWeightingScheme, leaderRotation,
                                                       type="continunous", faulty=True, numberOfViews=numberOfViews)

            if newLatencyWhenFaulty < bestLatencyWhenFaulty:
                bestLatencyWhenFaulty = newLatencyWhenFaulty
                bestWeightingScheme = nextWeightingScheme

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeightingScheme = nextWeightingScheme
            bestLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, nextWeightingScheme, leaderRotation,
                                                        type="continunous", faulty=True, numberOfViews=numberOfViews)

        # update the tempertaure and step counter
        temp = temp * (1 - theta)
        step += 1

    end = time.time()

    # print('--------------------------------')
    # print('-------- Simulated annealing --------')
    # print('--------------------------------')
    # print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    # print('Final solution latency: {} and latency under faulty conditions: {}'.format(bestLatency, bestLatencyWhenFaulty))
    # print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    # print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))

    return (bestLatency, bestLatencyWhenFaulty)

def weightedHotstuff(n, f, delta, leaderRotation=[], numberOfViews=1):
    # start the timer to register the convergence time
    start = time.time()

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting weighting assignment
    currentWeights = [1] * n
    for i in range(2 * f):
        currentWeights[i] = 1 + delta / f

    # starting with leader replica 0
    currentLeader = 0
    # get a baseline
    currentLatency = runWeightedHotstuff(n, f, delta, currentWeights, leaderRotation, type="best", numberOfViews=numberOfViews)
    currentLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, currentWeights, leaderRotation, type="best", faulty=True, numberOfViews=numberOfViews)

    # variables to retain the best solution found
    bestLeader = -1
    bestLatency = currentLatency
    bestLatencyWhenFaulty = currentLatencyWhenFaulty
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        newLeader = currentLeader

        while True:
            replicaFrom = random.randint(0, n - 1)
            if currentWeights[replicaFrom] == 1 + delta / f:
                break
        while True:
            replicaTo = random.randint(0, n - 1)
            if replicaTo != replicaFrom:
                break

        if replicaFrom == currentLeader:
            newLeader = replicaTo

        newWeights = currentWeights.copy()
        newWeights[replicaTo] = currentWeights[replicaFrom]
        newWeights[replicaFrom] = currentWeights[replicaTo]

        newLatency = runWeightedHotstuff(n, f, delta, newWeights, leaderRotation, type="best", numberOfViews=numberOfViews)

        if newLatency < currentLatency:
            currentLeader = newLeader
            currentWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentLeader = newLeader
                currentWeights = newWeights

        if newLatency == bestLatency:
            newLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, newWeights, leaderRotation, type="best",
                                                       faulty=True, numberOfViews=numberOfViews)

            if newLatencyWhenFaulty < bestLatencyWhenFaulty:
                bestLatencyWhenFaulty = newLatencyWhenFaulty
                bestWeights = newWeights

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeights = newWeights
            bestLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, newWeights, leaderRotation, type="best",
                                                        faulty=True, numberOfViews=numberOfViews)


        temp = temp * (1 - theta)
        step += 1

    end = time.time()

    # print('--------------------------------')
    # print('-------- Simulated annealing --------')
    # print('--------------------------------')
    # print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    # print('Final solution latency: {}'.format(bestLatency))
    # print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    # print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))
    # print(bestWeights)

    return (bestLatency, bestLatencyWhenFaulty)

def weightedHotstuffOptimalLeader(n, numberOfViews, faulty=False):
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
    currentLatency = runWeightedHotstuff(n, f, delta, awareWeights, type="weighted",
                                                                 leaderRotation=currentLeaderRotation, faulty=faulty, numberOfViews=numberOfViews)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the leader rotation scheme
        # swap two leaders
        nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring")

        newLatency = runWeightedHotstuff(n, f, delta, awareWeights, type="weighted",
                                                                 leaderRotation=nextLeaderRotation, faulty=faulty, numberOfViews=numberOfViews)
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

def generateLatenciesToLeader(n, leaderID, low, high):
    # latency induced by the distances between the leaderID replica and rest of replicas
    L = [0] * n

    for i in range(n):
        if i != leaderID:
            L[i] = random.randint(low, high)

    # for each type of message we add a transmission delay
    newview_delay = random.randint(0, 100)
    Lnew_view = [0] * n
    for i in range(n):
        if i != leaderID:
            Lnew_view[i] = L[i] + newview_delay

    prepare_delay = random.randint(0, 100)
    Lprepare = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprepare[i] = L[i] + prepare_delay

    precommit_delay = random.randint(0, 100)
    Lprecommit = [0] * n
    for i in range(n):
        if i != leaderID:
            Lprecommit[i] = L[i] + precommit_delay

    commit_delay = random.randint(0, 100)
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

f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas
leaderID = 0

awareWeights = [1] * n
for i in range(2 * f):
    awareWeights[i] = 1 + delta / f


# # generate the latencies for which we optimise the weighting scheme
# (Lnew_view, Lprepare, Lprecommit, Lcommit) = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

# HOTSTUFF with AWARE weighting scheme
print(runWeightedHotstuff(n, f, delta, [1] * n , [0], type="basic", numberOfViews=1))
print(runWeightedHotstuff(n, f, delta, awareWeights, [0], type="weighted", numberOfViews=1))
print(runWeightedHotstuff(n, f, delta, awareWeights, [0], faulty=True, type="weighted", numberOfViews=1))

print(weightedHotstuff(n, f, delta, [0],1))
print(continuousWeightedHotstuff(n, f, delta, [0],1))

# print(Lnew_view)
# print(Lprepare)
# print(Lprecommit)
# print(Lcommit)

# EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
basicLatency = []

weightedLatency = []
weightedFallback = []

bestLatency = []
bestFallback = []

continuousLatency = []
continuousFallback = []

bestLeaderLatency = []

viewNumbers = []
for i in range(1, 20):
    viewNumbers.append(i * 5)

for numberOfViews in viewNumbers:
    leaderRotation = getLeaderRotation(n, numberOfViews)
    # print(leaderRotation)
    # print(numberOfViews)

    # run in BASIC MODE
    latency = runWeightedHotstuff(n, f, delta, [1] * n , leaderRotation, type="basic", numberOfViews=numberOfViews) / numberOfViews
    basicLatency.append(latency)

    # run in WEIGHTED MODE
    latency = runWeightedHotstuff(n, f, delta, awareWeights , leaderRotation, type="weighted", numberOfViews=numberOfViews) / numberOfViews
    latencyFaulty = runWeightedHotstuff(n, f, delta, awareWeights, leaderRotation, type="weighted", faulty=True,
                                  numberOfViews=numberOfViews) / numberOfViews
    weightedLatency.append(latency)
    weightedFallback.append(latencyFaulty - latency)

    # run in BEST MODE
    (latency, latencyFaulty) = weightedHotstuff(n, f, delta, leaderRotation, numberOfViews)
    latency /= numberOfViews
    latencyFaulty /= numberOfViews
    bestLatency.append(latency)
    bestFallback.append(latencyFaulty - latency)

    # run in CONTINUOUS MODE
    (latency, latencyFaulty) = continuousWeightedHotstuff(n, f, delta, leaderRotation, numberOfViews)
    latency /= numberOfViews
    latencyFaulty /= numberOfViews
    continuousLatency.append(latency)
    continuousFallback.append(latencyFaulty - latency)

    # run in WEIGHTED MODE with LEADER OPTIMALITY
    latency = weightedHotstuffOptimalLeader(n, numberOfViews) / numberOfViews
    bestLeaderLatency.append(latency)


# Plot the Analysis
plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
         label='No Weights')
plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Randomly assigned AWARE Weights')

plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best assigned AWARE Weights')

plt.plot(viewNumbers, continuousLatency, color='red', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Continuous Weights')

plt.plot(viewNumbers, bestLeaderLatency, color='blue', marker='p', linestyle='--', linewidth=2, markersize=6,
         label='Best leader rotation on Randomly assigned AWARE Weights')

plt.title('Analysis of Average Latency per View in Hotstuff', fontsize=16)
plt.xlabel('Number of views', fontsize=14)
plt.ylabel('Average Latency per View [ms]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, weightedFallback, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Randomly assigned AWARE Weights')

plt.plot(viewNumbers, bestFallback, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best assigned AWARE Weights')

plt.plot(viewNumbers, continuousFallback, color='red', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Continuous Weights')

plt.title('Analysis of Fallback Latency Delay in Hotstuff', fontsize=16)
plt.xlabel('Number of views', fontsize=14)
plt.ylabel('Average Latency per View [ms]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
