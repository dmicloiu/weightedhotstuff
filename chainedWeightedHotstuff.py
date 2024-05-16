import random
import heapq
import matplotlib.pyplot as plt


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
    quorumWeight = 2 * f + 1  # quorum formation condition

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

## EXPERIMENT
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


# Graphical representation for BASIC MODE
plt.figure(figsize=(8, 6))
plt.hist(basicLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgBasicLatency, color='red', linestyle='--', label=f'Average Basic Latency: {avgBasicLatency:.2f}')
plt.title('Latency of Chained Hotstuff')
plt.xlabel('Latency')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

# Graphical representation for RANDOM MODE
plt.figure(figsize=(8, 6))
plt.hist(randomLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgRandomLatency, color='red', linestyle='--', label=f'Average Random Latency: {avgRandomLatency:.2f}')
plt.title('Latency of Weighted Chained Hotstuff with randomly assigned weights')
plt.xlabel('Latency')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

# Graphical representation for WEIGHTED MODE
plt.figure(figsize=(8, 6))
plt.hist(dynamicLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgDynamicLatency, color='red', linestyle='--', label=f'Average Dynamic Latency: {avgDynamicLatency:.2f}')
plt.title('Latency of Weighted Chained Hotstuff with dynamically assigned weights')
plt.xlabel('Latency')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()
