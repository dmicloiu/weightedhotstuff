import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import time, math
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
    if type == "weighted":
        for replicaIDx in range(2 * f):
            weights[replicaIDx] = vmax

        # the rest of the weights are already Vmin = 1

    # consider the case we are making f replicas faulty
    faulty_replicas = set()
    if type != "basic" and faulty:
        for replicaIdx in range(n):
            if len(faulty_replicas) < f and weights[replicaIdx] == vmax:
                faulty_replicas.add(replicaIdx)

    if type == "best":
        return runChainedHotstuffSimulatedAnnealing(n, quorumWeight, leaderRotation, faulty_replicas)

    return runChainedHotstuffSimulation(n, weights, quorumWeight, leaderRotation, faulty_replicas)


def runChainedHotstuffSimulation(n, weights, quorumWeight, leaderRotation, faulty_replicas={}):
    # keep track of the total latency prediction of the simulation
    latency = 0

    # queue of commands that we are currently processing
    currentProcessingCommands = []
    for viewNumber in range(numberOfViews):
        # in each view we start a new proposal -> hence add a new command that is currently processed
        currentProcessingCommands.append(viewNumber)

        # given that the Hotstuff algorithm uses 5 phases which use 4 quorum formations, when we get to 5
        # currently processing commands, we need to pop wince the first command in queue is executed and hence finished
        if len(currentProcessingCommands) == 5:
            currentProcessingCommands.pop()

        latency += processView(n, leaderRotation[viewNumber], currentProcessingCommands,
                               weights, quorumWeight, faulty_replicas)

    return latency


def processView(n, leaderID, currentProcessingCommands, weights, quorumWeight, faulty_replicas):
    # the total time it takes for performing the current phase for each command in the current view
    totalTime = 0
    for _ in currentProcessingCommands:
        # generate the latency vector of the leader -> latency of leader receiving the vote message from each replica
        Lphase = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

        # EXECUTE the current phase of the command -> leader waits for quorum formation with (n - f) messages from replicas
        totalTime += formQuorumChained(Lphase, weights, quorumWeight, faulty_replicas)

    return totalTime


def runChainedHotstuffSimulatedAnnealing(n, quorumWeight, leaderRotation, faulty_replicas={}):
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
    currentLatency = runChainedHotstuffSimulation(n, currentWeights, quorumWeight, leaderRotation, faulty_replicas)

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

        newLatency = runChainedHotstuffSimulation(n, currentWeights, quorumWeight, leaderRotation, faulty_replicas)

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


def generateLatenciesToLeader(n, leaderID, low, high):
    L = [0] * n

    for i in range(n):
        if i != leaderID:
            L[i] = random.randint(low, high)

    return L


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
    currentLatency = setupChainedHotstuffSimulation(n, currentLeaderRotation, type, faulty)

    bestLatency = currentLatency
    bestLeaderRotation = currentLeaderRotation

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        # generate "neighbouring" state for the leader rotation scheme
        # swap two leaders
        nextLeaderRotation = getLeaderRotation(n, numberOfViews, type="neighbouring")

        newLatency = setupChainedHotstuffSimulation(n, currentLeaderRotation, type, faulty)

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


f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

vmax = 1 + delta / f  # 2f replicas
vmin = 1  # n - 2f replicas

numberOfViews = 10

## EXPERIMENT 1
print("------------ EXPERIMENT 1 ------------")
simulations = 10000

avgBasicLatency = 0
avgWeightedLatency = 0

basicLatency = []
weightedLatency = []

for _ in range(simulations):
    leaderRotation = getLeaderRotation(n, numberOfViews)

    # run in BASIC MODE
    latency = setupChainedHotstuffSimulation(n, leaderRotation)
    basicLatency.append(latency)
    avgBasicLatency += (latency / simulations)

    # run in WEIGHTED MODE
    latency = setupChainedHotstuffSimulation(n, leaderRotation, type="weighted")
    weightedLatency.append(latency)
    avgWeightedLatency += (latency / simulations)


print(f"We perform Chained Hotstuff using {simulations} simulations of the protocol, using {numberOfViews} views.")
print(f"Average latency of Basic Chained Hotstuff: {avgBasicLatency}")
print(f"Average latency of Weighted Chained Hotstuff: {avgWeightedLatency}")

xlim_left = min(min(basicLatency), min(weightedLatency))
xlim_right = max(max(basicLatency), max(weightedLatency))

# DEBUG purposes
# print(xlim_left, xlim_right)

# Graphical representation for BASIC MODE
plt.figure(figsize=(8, 6))
plt.hist(basicLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgBasicLatency, color='red', linestyle='--', label=f'Average Basic Latency: {avgBasicLatency:.2f}')
plt.title('Latency of Chained Hotstuff')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.xlim([xlim_left, xlim_right])
plt.ylim([0, 700])
plt.legend()
plt.grid(True)
plt.show()

# Graphical representation for WEIGHTED MODE
plt.figure(figsize=(8, 6))
plt.hist(weightedLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgWeightedLatency, color='red', linestyle='--', label=f'Average Weighted Latency: {avgWeightedLatency:.2f}')
plt.title('Latency of Weighted Chained Hotstuff')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.xlim([xlim_left, xlim_right])
plt.ylim([0, 700])
plt.legend()
plt.grid(True)
plt.show()


# EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
basicLatency = []
weightedLatency = []
bestLatency = []
bestLeaderLatency = []

viewNumbers = []
for i in range(1, 40):
    viewNumbers.append(i * 5)

for numberOfViews in viewNumbers:
    leaderRotation = getLeaderRotation(n, numberOfViews)

    # run in BASIC MODE
    latency = setupChainedHotstuffSimulation(n, leaderRotation) / numberOfViews
    basicLatency.append(latency)

    # run in WEIGHTED MODE
    latency = setupChainedHotstuffSimulation(n, leaderRotation, type="weighted") / numberOfViews
    weightedLatency.append(latency)

    # run in BEST MODE
    latency = setupChainedHotstuffSimulation(n, leaderRotation, type="best") / numberOfViews
    bestLatency.append(latency)

    # run in WEIGHTED MODE with leader rotation optimisation
    latency = chainedHotstuffOptimalLeader(n, numberOfViews, type="weighted") / numberOfViews
    bestLeaderLatency.append(latency)

# Plot the Analysis
plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
         label='No Weights')
plt.plot(viewNumbers, weightedLatency, color='red', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Randomly assigned AWARE Weights')
plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best assigned AWARE Weights')
plt.plot(viewNumbers, bestLeaderLatency, color='orange', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Best leader rotation on Randomly assigned AWARE Weights')

plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('Number of views', fontsize=14)
plt.ylabel('Average Latency per View [ms]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# EXPERIMENT 3 -> faulty nodes
print("------------ EXPERIMENT 3 ------------")

numberOfViews = 10
leaderRotation = getLeaderRotation(n, numberOfViews)

basicLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='basic')
weightedLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=False)
weightedLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='weighted', faulty=True)
bestLatency = setupChainedHotstuffSimulation(n, leaderRotation, type='best')
bestLatencyFaulty = setupChainedHotstuffSimulation(n, leaderRotation, type='best', faulty = True)
weightedLatencyBestLeader = chainedHotstuffOptimalLeader(n, numberOfViews, type='weighted')
bestLatencyBestLeader = chainedHotstuffOptimalLeader(n, numberOfViews, type="best")

print(f"Simulation of Chained Hotstuff over {numberOfViews} views")
print("---------------")
print(f"No weights {basicLatency}")
print("---------------")
print(f"AWARE Weighted Scheme {weightedLatency}")
print(f"AWARE Weighted Scheme FAULTY {weightedLatencyFaulty}")
print("---------------")
print(f"Simulated Annealing (Best) Weighting Scheme {bestLatency}")
print(f"Simulated Annealing (Best) Weighting Scheme FAULTY {bestLatencyFaulty}")
print("---------------")
print(f"AWARE Weighted Scheme - BEST leader rotation {weightedLatencyBestLeader}")
print(f"Simulated Annealing (Best) Weighting Scheme - BEST leader rotation {bestLatencyBestLeader}")
print("---------------")

results = {
    "Basic": basicLatency,
    "Weighted (Non-faulty)": weightedLatency,
    "Weighted (Best leader rotation)": weightedLatencyBestLeader,
    "Weighted (Faulty)": weightedLatencyFaulty,
    "Best (Non-faulty)": bestLatency,
    "Best (Best leader rotation)": bestLatencyBestLeader,
    "Best (Faulty)": bestLatencyFaulty
}

simulation_types = list(results.keys())
simulation_results = list(results.values())

plt.figure(figsize=(10, 8))
bars = plt.barh(simulation_types, simulation_results, color=['skyblue', 'skyblue', 'green', 'red', 'skyblue', 'green', 'red'])
plt.xlabel('Latency [ms]', fontsize=14)
plt.ylabel('Weighting Assignment', fontsize=14)
plt.title(f'Weighted Chained Hotstuff performance over {numberOfViews} views', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding values beside the bars
for i, result in enumerate(simulation_results):
    plt.text(result + 0.1, i, str(result), va='center', fontsize=10, style="oblique")

plt.gca().invert_yaxis()  # Invert y-axis to display simulation types from top to bottom
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
