import time, random, math
import numpy as np
import heapq
from experimental_utils import getLeaderRotation, generateLatenciesToLeader

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

    return quorum_weight


def formQuorumWeighted(n, LMessageReceived, weights, quorumWeight, faulty_replicas={}):
    heap = []
    for replicaIdx in range(n):
        if replicaIdx not in faulty_replicas:
            heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    replicasUsed = 0
    while weight < quorumWeight:
        # special case since we work with double numbers
        if len(heap) == 0 and abs(quorumWeight - weight) <= 1e-6:
            break

        (receivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Hotstuff all relicas have the same weight
        replicasUsed += 1
        agreementTime = receivingTime

    return agreementTime


def predictLatency(n, networkTopology, Lphases, weights, quorumWeight, type="basic", leaderRotation=[], faulty_replicas={}, numberOfViews=1):
    latency = 0
    # predict latency for Hotstuff algorithm run with numberOfViews rounds
    for viewNumber in range(numberOfViews):
        (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]

        if type == "optimalLeader":
            (Lnew_view, Lprepare, Lprecommit, Lcommit) = generateLatenciesToLeader(n, leaderRotation[viewNumber], networkTopology)

        # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
        tPREPARE = formQuorumWeighted(n, Lnew_view, weights, quorumWeight, faulty_replicas)

        # after quorum formation -> leader sends PREPARE messages

        # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
        tPRECOMIT = formQuorumWeighted(n, Lprepare, weights, quorumWeight, faulty_replicas)

        # after quorum formation -> leader sends PRE-COMMIT messages

        # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
        tCOMMIT = formQuorumWeighted(n, Lprecommit, weights, quorumWeight, faulty_replicas)

        # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
        tDECIDE = formQuorumWeighted(n, Lcommit, weights, quorumWeight, faulty_replicas)

        latency += (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)

    ## total time of a Hotstuff run with numberOfViews rounds
    return latency


# there are 5 behaviours implemented for Hotstuff
# basic -> all weights are 1, hence it emulates the classic hotstuff behaviour
# weighted -> AWARE weighting scheme Hotstuff - first 2f are Vmax the rest are Vmin
# best -> AWARE weighting scheme Hotstuff - weights distributed with Simulated Annealing to minimise latency
# optimalLeader -> AWARE weighting scheme Hotstuff - Simulated Annealing approach to finding the best leaderRotation
# continuous -> continuous weighting scheme Hotstuff - Simulated Annealing approach to finding the best weighting scheme
def runWeightedHotstuff(n, f, delta, networkTopology, Lphases, weightingScheme, leaderRotation=[], type="basic", faulty=False, numberOfViews=1):
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition -> majority for basic weighting scheme

    if type == "weighted" or type == "best" or type == "optimalLeader":
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

    return predictLatency(n, networkTopology, Lphases, weightingScheme, quorumWeight, type=type,
                          leaderRotation=leaderRotation, faulty_replicas=faulty_replicas, numberOfViews=numberOfViews)


def continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation=[], numberOfViews=1):
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
    currentLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeightingScheme, leaderRotation,
                                         type="continunous", numberOfViews=numberOfViews)
    currentLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeightingScheme, leaderRotation,
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
        newLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, nextWeightingScheme, leaderRotation,
                                         type="continunous", numberOfViews=numberOfViews)

        # print(nextWeightingScheme)

        # if it performs better
        if newLatency < currentLatency:
            currentWeightingScheme = nextWeightingScheme
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeightingScheme = nextWeightingScheme

        if newLatency == bestLatency:
            newLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, nextWeightingScheme, leaderRotation,
                                                       type="continunous", faulty=True, numberOfViews=numberOfViews)

            if newLatencyWhenFaulty < bestLatencyWhenFaulty:
                bestLatencyWhenFaulty = newLatencyWhenFaulty
                bestWeightingScheme = nextWeightingScheme

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeightingScheme = nextWeightingScheme
            bestLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, nextWeightingScheme, leaderRotation,
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
    # print(bestWeightingScheme)

    return (bestLatency, bestLatencyWhenFaulty)


def weightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation=[], numberOfViews=1):
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
    currentLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeights, leaderRotation, type="best",
                                         numberOfViews=numberOfViews)
    currentLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeights, leaderRotation, type="best",
                                                   faulty=True, numberOfViews=numberOfViews)

    # variables to retain the best solution found
    bestLeader = -1
    bestLatency = currentLatency
    bestLatencyWhenFaulty = currentLatencyWhenFaulty
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        vmin_replicas = []
        vmax_replicas = []

        for replicaID, weight in enumerate(currentWeights):
            if weight == 1 + delta / f:
                vmax_replicas.append(replicaID)
            else:
                vmin_replicas.append(replicaID)

        maxReplica = np.random.choice(vmax_replicas)
        minReplica = np.random.choice(vmin_replicas)

        newWeights = currentWeights.copy()
        newWeights[maxReplica] = currentWeights[minReplica]
        newWeights[minReplica] = currentWeights[maxReplica]

        newLatency = runWeightedHotstuff(n, f, delta,  networkTopology, Lphases, newWeights, leaderRotation, type="best",
                                         numberOfViews=numberOfViews)

        if newLatency < currentLatency:
            currentWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeights = newWeights

        if newLatency == bestLatency:
            newLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, newWeights, leaderRotation, type="best",
                                                       faulty=True, numberOfViews=numberOfViews)

            if newLatencyWhenFaulty < bestLatencyWhenFaulty:
                bestLatencyWhenFaulty = newLatencyWhenFaulty
                bestWeights = newWeights

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeights = newWeights
            bestLatencyWhenFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, newWeights, leaderRotation, type="best",
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

    return (bestLatency, bestLatencyWhenFaulty)


def weightedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, awareWeights, numberOfViews, faulty=False):
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
    currentLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, type="optimalLeader",
                                         leaderRotation=currentLeaderRotation, faulty=faulty,
                                         numberOfViews=numberOfViews)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the leader rotation scheme
        # swap two leaders
        nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring", currentLeaderRotation=currentLeaderRotation)

        newLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, type="optimalLeader",
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

def weightedBestHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, awareWeights, numberOfViews, faulty=False):
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
    # starting weighting scheme -> AWARE scheme
    currentWeights = awareWeights

    # get a baseline
    currentLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeights, type="optimalLeader",
                                         leaderRotation=currentLeaderRotation, faulty=faulty,
                                         numberOfViews=numberOfViews)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation
    bestWeights = currentWeights

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # choose what we optimise in this step
        choice = 0 # optimise for leader
        if random.random() < 0.5:
            choice = 1 # optimise for weights

        if choice == 0:
            # generate "neighbouring" state for the leader rotation scheme
            # swap two leaders
            nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring", currentLeaderRotation=currentLeaderRotation)

            newLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, currentWeights, type="optimalLeader",
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

        else:
            while True:
                replicaFrom = random.randint(0, n - 1)
                if currentWeights[replicaFrom] == 1 + delta / f:
                    break
            while True:
                replicaTo = random.randint(0, n - 1)
                if replicaTo != replicaFrom:
                    break


            newWeights = currentWeights.copy()
            newWeights[replicaTo] = currentWeights[replicaFrom]
            newWeights[replicaFrom] = currentWeights[replicaTo]

            newLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, newWeights, currentLeaderRotation, type="best",
                                             numberOfViews=numberOfViews)

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
    # print(bestLeaderRotation)
    # print(bestWeights)
    return bestLatency
