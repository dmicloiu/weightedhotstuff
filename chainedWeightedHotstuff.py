import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import time, math


# use the same function for Quorum Formation on both basic and weighted Chained Hotstuff
# we are using weights equal to 1 for the normal version
def formQuorumChained(LMessageReceived, weights, quorumWeight):
    heap = []
    for replicaIdx in range(n):
        heapq.heappush(heap, (LMessageReceived[replicaIdx], weights[replicaIdx]))

    weight = 0
    agreementTime = 0
    while weight < quorumWeight:
        (recivingTime, weightOfVote) = heapq.heappop(heap)
        weight += weightOfVote  ## in Basic Chained Hotstuff all relicas have the same weight
        agreementTime = recivingTime

    return agreementTime


def runChainedHotstuffSimulation(n, numberOfViews, type="basic"):
    # randomly choose the leaders for each new view in the Chained-Hotstuff simulation
    leaderRotation = []
    for _ in range(numberOfViews):
        leaderRotation.append(random.randint(0, n - 1))

    # in the beginning we have to information on the performance of replicas in the previous view
    # -> all replicas perform as good as possible
    replicaPerformance = [0] * n

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

        (latencyOfView, replicaPerformance) = processView(n, f, leaderRotation[viewNumber], currentProcessingCommands,
                                                          type, replicaPerformance)
        latency += latencyOfView

    return latency


def processView(n, f, leaderID, currentProcessingCommands, type, replicaPerformance):
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition

    if type != "basic":
        quorumWeight = 2 * (f + delta) + 1

    # basic means normal Chained Hotstuff -> all weights are 1
    # random means weighted Chained Hotstuff with the weights assigned randomly
    # dynamic means weighted Chained Hotstuff with weights assigned based on replica performance heuristic based on previous completed view

    # basic weighting scheme
    weights = [1] * n
    if type == "random":
        # the rest of the weights are already Vmin = 1
        # leader should have Vmax weight
        weights[leaderID] = vmax
        replicaID = leaderID

        # keep track how many replicas are assigned Vmax sice only 2f ones should be
        numberOfVmaxReplicas = 1
        while True:
            while weights[replicaID] == vmax:
                replicaID = random.randint(0, n - 1)

            # assign the new randomly chosen replica Vmax weight
            weights[replicaID] = vmax
            numberOfVmaxReplicas += 1

            # stop assigning Vmax replicas when we reach 2*f replicas
            if (numberOfVmaxReplicas == 2 * f):
                break

    elif type == "dynamic":
        # leader should have Vmax weight
        weights[leaderID] = vmax

        heap = []
        for replicaIdx in range(n):
            heapq.heappush(heap, (replicaPerformance[replicaIdx], replicaIdx))

        # keep track how many replicas are assigned Vmax sice only 2f ones should be
        numberOfVmaxReplicas = 1
        while True:
            (latency, replicaID) = heapq.heappop(heap)

            if replicaID == leaderID:
                continue

            weights[replicaID] = vmax
            numberOfVmaxReplicas += 1

            # stop assigning Vmax replicas when we reach 2*f replicas
            if (numberOfVmaxReplicas == 2 * f):
                break

        # the rest of the weights are already Vmin = 1

    # type best indicates that we use simulated annealing to get the best performance
    if type == "best":
        return processView_Simulated_Annealing(n, f, leaderID, currentProcessingCommands, quorumWeight)

    # the total time it takes for performing the current phase for each command in the current view
    totalTime = 0
    avgLatency = [0] * n
    for _ in currentProcessingCommands:
        # generate the latency vector of the leader -> latency of leader receiving the vote message from each replica
        Lphase = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

        avgLatency += Lphase

        # EXECUTE the current phase of the command -> leader waits for quorum formation with (n - f) messages from replicas
        totalTime += formQuorumChained(Lphase, weights, quorumWeight)

    numberOfProcessingCommands = len(currentProcessingCommands)
    for idx in range(n):
        avgLatency[idx] /= numberOfProcessingCommands

    return (totalTime, avgLatency)

def processView_Simulated_Annealing(n, f, leaderID, currentProcessingCommands, quorumWeight):
    Lphases = []

    for _ in currentProcessingCommands:
        # generate the latency vector of the leader -> latency of leader receiving the vote message from each replica
        Lphase = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
        Lphases.append(Lphase)

    # for assessing the simulated annealing process
    start = time.time()

    # declare a seed for this process
    random.seed(300)

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
    (currentLatency, avgLatency) = predictLatencySimulatedAnnealing(n, currentWeights, Lphases, quorumWeight)
    bestLatency = currentLatency
    bestAvgLatency = avgLatency
    bestWeights = []

    # for monitoring purposes of the simulating annealing
    jumps = 0

    while step < step_max and temp > t_min:
        while True:
            replicaFrom = random.randint(0, n - 1)
            if currentWeights[replicaFrom] == vmax and replicaFrom != leaderID:
                break

        while True:
            replicaTo = random.randint(0, n - 1)
            if replicaTo != replicaFrom:
                break

        newWeights = currentWeights.copy()
        newWeights[replicaTo] = currentWeights[replicaFrom]
        newWeights[replicaFrom] = currentWeights[replicaTo]

        (newLatency, newAvgLatency) = predictLatencySimulatedAnnealing(n, currentWeights, Lphases, quorumWeight)

        if newLatency < currentLatency:
            currentWeights = newWeights
        else:
            rand = random.uniform(0, 1)
            if rand < math.exp(-(newLatency - currentLatency) / temp):
                jumps = jumps + 1
                currentWeights = newWeights

        if newLatency < bestLatency:
            bestLatency = newLatency
            bestAvgLatency = newAvgLatency
            bestWeights = newWeights

        temp = temp * (1 - theta)
        step += 1

    # DEBUG purposes
    # print(bestWeights)
    return (bestLatency, bestAvgLatency)

def predictLatencySimulatedAnnealing(n, currentWeights, Lphases, quorumWeight):
    # the total time it takes for performing the current phase for each command in the current view
    totalTime = 0
    avgLatency = [0] * n

    for Lphase in Lphases:
        avgLatency += Lphase

        # EXECUTE the current phase of the command -> leader waits for quorum formation with (n - f) messages from replicas
        totalTime += formQuorumChained(Lphase, currentWeights, quorumWeight)

    numberOfProcessingCommands = len(Lphase)
    for idx in range(n):
        avgLatency[idx] /= numberOfProcessingCommands

    return (totalTime, avgLatency)

def generateLatenciesToLeader(n, leaderID, low, high):
    L = [0] * n

    for i in range(n):
        if i != leaderID:
            L[i] = random.randint(low, high)

    return L


f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

vmax = 1 + delta / f  # 2f replicas
vmin = 1  # n - 2f replicas

numberOfViews = 10

## EXPERIMENT 1
simulations = 10000

avgBasicLatency = 0
avgRandomLatency = 0
avgDynamicLatency = 0

basicLatency = []
randomLatency = []
dynamicLatency = []

for _ in range(simulations):
    # run in BASIC MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews)
    basicLatency.append(latency)
    avgBasicLatency += (latency / simulations)

    # run in RANDOM MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews, type="random")
    randomLatency.append(latency)
    avgRandomLatency += (latency / simulations)

    # run in DYNAMIC MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews, type="dynamic")
    dynamicLatency.append(latency)
    avgDynamicLatency += (latency / simulations)


print(f"We perform Chained Hotstuff using {simulations} simulations of the protocol, using {numberOfViews} views.")
print(f"Average latency of Basic Chained Hotstuff: {avgBasicLatency}")
print(f"Average latency of Weighted Chained Hotstuff - randomly assigned weights: {avgRandomLatency}")
print(f"Average latency of Weighted Chained Hotstuff - dynamically assigned weights: {avgDynamicLatency}")

xlim_left = min(min(randomLatency), min(min(basicLatency), min(dynamicLatency)))
xlim_right = max(max(randomLatency), max(max(basicLatency), max(dynamicLatency)))

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

# Graphical representation for RANDOM MODE
plt.figure(figsize=(8, 6))
plt.hist(randomLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgRandomLatency, color='red', linestyle='--', label=f'Average Random Latency: {avgRandomLatency:.2f}')
plt.title('Latency of Weighted Chained Hotstuff with randomly assigned weights')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.xlim([xlim_left, xlim_right])
plt.ylim([0, 700])
plt.legend()
plt.grid(True)
plt.show()

# Graphical representation for DYNAMIC MODE
plt.figure(figsize=(8, 6))
plt.hist(dynamicLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgDynamicLatency, color='red', linestyle='--', label=f'Average Dynamic Latency: {avgDynamicLatency:.2f}')
plt.title('Latency of Weighted Chained Hotstuff with dynamically assigned weights')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.xlim([xlim_left, xlim_right])
plt.ylim([0, 700])
plt.legend()
plt.grid(True)
plt.show()


## EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
basicLatency = []
randomLatency = []
dynamicLatency = []
bestLatency = []

viewNumbers = []
for i in range(1, 40):
    viewNumbers.append(i * 5)


for numberOfViews in viewNumbers:
    # run in BASIC MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews) / numberOfViews
    basicLatency.append(latency)

    # run in RANDOM MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews, type="random") / numberOfViews
    randomLatency.append(latency)

    # run in DYNAMIC MODE
    latency = runChainedHotstuffSimulation(n, numberOfViews, type="dynamic") / numberOfViews
    dynamicLatency.append(latency)

    # run in BEST mode
    latency = runChainedHotstuffSimulation(n, numberOfViews, type="best") / numberOfViews
    bestLatency.append(latency)

# Plot the Analysis
plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6, label='Egalitarian Weights')
plt.plot(viewNumbers, randomLatency, color='red', marker='s', linestyle='--', linewidth=2, markersize=6, label='Randomly Assigned Binary Weights')
plt.plot(viewNumbers, dynamicLatency, color='orange', marker='^', linestyle='-.', linewidth=2, markersize=6, label='Dynamically Assigned Binary Weights')
plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6, label='Best Assigned Binary Weights')

plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('Number of views', fontsize=14)
plt.ylabel('Average Latency per View [ms]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

