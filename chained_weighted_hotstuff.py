import heapq
import time, math
from collections import deque
from experimental_utils import *


# use the same function for quorum formation on both basic and weighted Chained Hotstuff
# we are using weights equal to 1 for the normal version
def formQuorumChained(n, LMessageReceived, weights, quorumWeight, faulty_replicas={}):
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

# there are 4 behaviours implemented for Chained Hotstuff
# basic -> classic Chained Hotstuff behaviour (all weights are 1)
# weighted -> just weighted version - f best connected and f worst connected are Vmax
# best -> Best Assigned
# weighted-leader -> Optimal Leader Rotation
# best-leader -> (Optimal Leader Rotation + Best Assigned)
def setupChainedHotstuffSimulation(n, f, delta, networkToplogy, Lphases, awareWeights, leaderRotation,
                                   type="basic", numberOfViews=1, faulty=False):
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
        return runChainedHotstuffSimulatedAnnealing(n, f, delta, type, networkToplogy, Lphases, quorumWeight, leaderRotation, numberOfViews,
                                                    faulty_replicas)

    return runChainedHotstuffSimulation(n, type, networkToplogy, Lphases, weights, quorumWeight, leaderRotation, numberOfViews,
                                        faulty_replicas)

# function for predicting the latency of a protocol run (emulates Weighted Chained Hotstuff behaviour)
# adapted from AWARE's deterministic latency prediction algorithm -> self-monitoring
def runChainedHotstuffSimulation(n, type, networkTopology, Lphases, weights, quorumWeight, leaderRotation, numberOfViews,
                                 faulty_replicas={}):
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

        latency += processView(n, type, networkTopology, Lphases, viewNumber, leaderRotation[viewNumber],
                               currentProcessingCommands,
                               weights, quorumWeight, faulty_replicas)

    return latency

# function for processing a view run in Weighted Chained Hotstuff
def processView(n, type, networkTopology, Lphases, viewNumber, leaderID, currentProcessingCommands, weights,
                quorumWeight, faulty_replicas={}):
    # the total time it takes for performing the current phase for each command in the current view
    totalTime = 0

    for blockProposalNumber in currentProcessingCommands:
        # get the latency vector of the leader -> latency of leader receiving the vote message from each replica
        Lphase = Lphases[blockProposalNumber][viewNumber - blockProposalNumber]

        # if we optimise leader rotation -> latency vectors reported by the leader need to be generated
        if type == "best-leader" or type == "weighted-leader" or type == "basic-leader":
            Lphase = generateLatenciesToLeader(n, leaderID, networkTopology)[viewNumber - blockProposalNumber]

        # EXECUTE the current phase of the command -> leader waits for quorum formation with (n - f) messages from replicas
        totalTime += formQuorumChained(n, Lphase, weights, quorumWeight, faulty_replicas)

    return totalTime

# Best Assigned Weighted Chained Hotstuff
def runChainedHotstuffSimulatedAnnealing(n, f, delta, type, networkTopology, Lphases, quorumWeight, leaderRotation, numberOfViews,
                                         faulty_replicas={}):
    # for assessing the simulated annealing process
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

    # get a baseline
    currentLatency = runChainedHotstuffSimulation(n, type, networkTopology, Lphases, currentWeights, quorumWeight,
                                                  leaderRotation, numberOfViews, faulty_replicas)

    bestLatency = currentLatency
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
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

        newLatency = runChainedHotstuffSimulation(n, type, networkTopology, Lphases, currentWeights, quorumWeight,
                                                  leaderRotation, numberOfViews, faulty_replicas)

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

# Optimal Leader Rotation Weighted Chained Hotstuff
def chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, awareWeights, numberOfViews, type='basic', faulty=False):
    # for assessing the simulated annealing process
    start = time.time()

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2

    # starting leader rotation -> basic "round-robin"
    currentLeaderRotation = getLeaderRotation(n, numberOfViews)

    # get a baseline
    currentLatency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, currentLeaderRotation,
                                                    type + "-leader", numberOfViews, faulty)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the leader rotation scheme
        # swap two leaders
        nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring")
        newLatency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, currentLeaderRotation,
                                                    type + "-leader", numberOfViews, faulty)

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
    # print(bestLeaderRotation)
    return bestLatency

# (Optimal Leader Rotation + Best Assigned) Weighted Chained Hotstuff
def chainedHotstuffBestAndOptimalLeader(n, f, delta, networkTopology, Lphases, numberOfViews):
    # for assessing the simulated annealing process
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

    quorumWeight = 2 * (f + delta) + 1

    # starting leader rotation -> basic "round robin"
    currentLeaderRotation = getLeaderRotation(n, numberOfViews)

    # get a baseline
    currentLatency = runChainedHotstuffSimulation(n,"best-leader", networkTopology, Lphases, currentWeights,
                                                  quorumWeight, currentLeaderRotation, numberOfViews)

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
            newLatency = runChainedHotstuffSimulation(n,"best-leader", networkTopology, Lphases, currentWeights, quorumWeight,
                                                      nextLeaderRotation, numberOfViews)

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

            newLatency = runChainedHotstuffSimulation(n,"best-leader",  networkTopology, Lphases, newWeights,
                                                      quorumWeight, currentLeaderRotation, numberOfViews)

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
    return bestLatency