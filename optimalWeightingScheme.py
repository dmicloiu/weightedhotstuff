import time, random, math
import heapq


def all_subsets(index, n, current_subsets):
    if index == n:
        return current_subsets

    if index == 0:
        for i in range(n):
            current_subsets.append([i])
    else:
        for subset in current_subsets:
            last_added = subset[-1]
            for i in range(last_added + 1, n):
                current_subsets.append(subset + [i])

    return all_subsets(index + 1, n, current_subsets)


# function for computing the quorum weight such as the conditions for quorums are satisfied
def compute_quorum_weight(n, f, delta, weights):
    heap = []
    for replicaIdx in range(n):
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
    # check that they overlap by at least (f + 1) nodes
    possible_quorums = all_subsets(0, n, [])

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


def formQuorumWeighted(LMessageReceived, weights, quorumWeight, faulty=False):
    heap = []
    for replicaIdx in range(n):
        heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    removed = False
    while weight < quorumWeight:
        # special case since we work with double numbers
        if len(heap) == 0 and abs(quorumWeight - weight) <= 1e-6:
            # prints for DEBUG purposes
            # print(quorumWeight)
            # print(weight)
            break

        # if the replica is faulty we treat worst case scenario the 2nd best performing one is failing (first best is leader)
        if faulty and not removed:
            (receivingTime, weightOfVote) = heapq.heappop(heap)

            # if leader -> not faulty
            if receivingTime != 0:
                removed = True
            else:
                weight += weightOfVote  ## in Basic Hotstuff all relicas have the same weight
                agreementTime = receivingTime
            continue

        (receivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Hotstuff all relicas have the same weight
        agreementTime = receivingTime

    return agreementTime


def predictLatency(n, f, delta, weights, faulty=False):
    quorumWeight = compute_quorum_weight(n, f, delta, weights)

    # if the current weighting scheme violets AVAILABILITY and/or CONSISTENCY
    if quorumWeight == -1:
        # return a very big number so that the configuration is discarded
        return 1e6

    # predict latency for Hotstuff algorithm
    # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
    tPREPARE = formQuorumWeighted(Lnew_view, weights, quorumWeight, faulty)

    # after quorum formation -> leader sends PREPARE messages

    # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
    tPRECOMIT = formQuorumWeighted(Lprepare, weights, quorumWeight, faulty)

    # after quorum formation -> leader sends PRE-COMMIT messages

    # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
    tCOMMIT = formQuorumWeighted(Lprecommit, weights, quorumWeight, faulty)

    # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
    tDECIDE = formQuorumWeighted(Lcommit, weights, quorumWeight, faulty)

    print(quorumWeight)
    print(weights)
    print(faulty)
    print((tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE))
    print("----------------------")

    ## total time of a Hotstuff view run
    return (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)


def simulated_annealing(n, f, delta):
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
    currentLatency = predictLatency(n, f, delta, currentWeightingScheme)
    currentLatencyWhenFaulty = predictLatency(n, f, delta, currentWeightingScheme, faulty=True)

    bestLatency = currentLatency
    bestLatencyWhenFaulty = currentLatencyWhenFaulty
    bestWeightingScheme = []

    while step < step_max and temp > t_min:
        # # generate "neighbouring" state for the weighting scheme
        # nextPerturbation = [random.uniform(currentWeightingScheme[idx] - 1, perturbation_step) for idx in range(n)]
        #
        # # update the weighting scheme
        # nextWeightingScheme = [1 + nextPerturbation[i] for i in range(n)]
        nextWeightingScheme = []
        for i in range(n):
            lowerBound = max(currentWeightingScheme[i] - 0.1, 0)
            upperBound = min(currentWeightingScheme[i] + 0.1, 2)
            nextWeightingScheme.append(random.uniform(lowerBound, upperBound))

        # compute the energy of the new state
        newLatency = predictLatency(n, f, delta, nextWeightingScheme)

        # if it performs better
        if newLatency < currentLatency:
            currentWeightingScheme = nextWeightingScheme
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeightingScheme = nextWeightingScheme

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestWeightingScheme = nextWeightingScheme

        if newLatency == bestLatency:
            newLatencyWhenFaulty = predictLatency(n, f, delta, nextWeightingScheme, True)

            if newLatencyWhenFaulty < bestLatencyWhenFaulty:
                bestLatencyWhenFaulty = newLatencyWhenFaulty
                bestWeightingScheme = nextWeightingScheme

        # update the tempertaure and step counter
        temp = temp * (1 - theta)
        step += 1

    end = time.time()

    print('--------------------------------')
    print('-------- Simulated annealing --------')
    print('--------------------------------')
    print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    print(
        'Final solution latency: {} and latency under faulty conditions: {}'.format(bestLatency, bestLatencyWhenFaulty))
    print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))

    return bestWeightingScheme


def generateLatenciesToLeader(n, leaderID, low, high):
    L = [0] * n

    for i in range(n):
        if i != leaderID:
            L[i] = random.randint(low, high)

    return sorted(L)


f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

leaderID = 0

# generate the latencies for which we optimise the weighting scheme
Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

# HOTSTUFF with AWARE weighting scheme
predictLatency(n, f, delta, [2, 2, 1, 1, 1], faulty=False)
predictLatency(n, f, delta, [2, 2, 1, 1, 1], faulty=True)

print(simulated_annealing(n, f, delta))
