import random
import heapq
import time
import math
import matplotlib.pyplot as plt


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
        weight += 1  ## in Basic Hotstuff all relicas have the same weight
        replicaIdx += 1
        agreementTime = LMessageReceived[replicaIdx]

    return agreementTime


def predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit):
    quorumWeight = 2 * f + 1  # quorum formation condition

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


def generateLatenciesToLeader(n, leaderID, low, high):
    L = [0] * n

    for i in range(n):
        if i != leaderID:
            L[i] = random.randint(low, high)

    return L

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

    # declare a seed for this process
    random.seed(500)

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

    print('--------------------------------')
    print('--------' + suffix + ' Simulated annealing')
    print('--------------------------------')
    print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    print('Final solution latency:', bestLat)
    print('Best Configuration:  R_max: {}, weight: {} | R_min: {}, weight: {} with leader {}'.format(r_max, vmax, r_min,
                                                                                                     vmin, bestLeader))
    print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))



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

### EXPERIMENT 1
print("------------ EXPERIMENT 1 ------------")
Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)

print(f"Basic Hotstuff with leader {leaderID} completes a view in {basicLatency}.")
print(f"Weighted Hotstuff with leader {leaderID} completes a view in {weightedLatency}.")
print("\n")

### EXPERIMENT 2
print("------------ EXPERIMENT 2 ------------")
simulations = 10000

averageDifference = 0
timesBasicIsFaster = 0
timesEqualPerformance = 0

differences = []
for _ in range(simulations):
    Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

    basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
    weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)

    if basicLatency < weightedLatency:
        timesBasicIsFaster += 1
    elif basicLatency == weightedLatency:
        timesEqualPerformance += 1
    else:
        differences.append(basicLatency - weightedLatency)
        averageDifference += basicLatency - weightedLatency

# compute the average
averageDifference /= simulations

print(f"Basic Hotstuff is faster than the weighted version in {timesBasicIsFaster} view simulations.")
print(f"The two algorithms have equal performance in {timesEqualPerformance} view simulations, accounting for {timesEqualPerformance * 100 / simulations}% of simulations.")
print(f"Weighted Hotstuff is on average with {averageDifference} faster than the Basic version.")
print("\n")

# debug purposes
# print(differences)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(differences, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=averageDifference, color='red', linestyle='--', label=f'Average Difference: {averageDifference:.2f}')
plt.title('Difference in Latency Basic vs Weighted Hotstuff')
plt.xlabel('Difference')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()


### EXPERIMENT 3
print("------------ EXPERIMENT 3 ------------")

numberOfViews = 10
leaderRotation = []
for _ in range(numberOfViews):
    leaderRotation.append(random.randint(0, n - 1))

print(f"The experiment is conducted using {numberOfViews} views and the following leader rotation scheme {leaderRotation}.")

basicLatency = 0
weightedLatency = 0
for viewNumber in range(numberOfViews):
    leaderID = leaderRotation[viewNumber]

    weightsOfView = []
    for i in range(2 * f):
        weightsOfView.append(vmax)
    for i in range(n - 2 * f):
        weightsOfView.append(vmin)

    # the leader has to have vmax
    if weightsOfView[leaderID] == vmin:
        weightsOfView[leaderID] = vmax
        ### since first 2f replicas have weight vmax is guaranteed that replica 0 has max weight
        weightsOfView[0] = vmin

    # construct latency vectors retained by the leader
    Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

    basicLatency += predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
    weightedLatency += predictLatencyWeightedHotstuff(n, f, weightsOfView, Lnew_view, Lprepare, Lprecommit, Lcommit)


print(f"Basic Hotstuff has on average {basicLatency / numberOfViews} latency over the simulated views.")
print(f"Weighted Hotstuff has on average {weightedLatency / numberOfViews} latency over the simulated views.")
print("\n")


### EXPERIMENT 4
print("------------ EXPERIMENT 4 ------------")

# generate the latency vectors - network setup for which we optimise the weighting assignment
Lnew_view = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
Lprepare = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
Lprecommit = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)
Lcommit = generateLatenciesToLeader(n, leaderID=0, low=0, high=1000)

basicLatency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
weightedLatency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)

print(f"Basic Hotstuff yields latency of {basicLatency}.")
print(f"Weighted Hotstuff yields latency of {weightedLatency}.")
print("The performance of the simulated annealing weighted assignment for the given network setup:\n")
simulated_annealing(n, f, delta, Lnew_view, Lprepare, Lprecommit, Lcommit, suffix='')
print("\n")

### EXPERIMENT 5
print("------------ EXPERIMENT 5 ------------")

simulations = 10000
averageBasicLatency = 0
basicLatencies = []
for _ in range(simulations):
    Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

    latency = predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
    averageBasicLatency += latency
    basicLatencies.append(latency)

averageBasicLatency /= simulations

print(f"The avergae Basic Hotstuff Latency is {averageBasicLatency}.\n")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(basicLatencies, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=averageBasicLatency, color='red', linestyle='--', label=f'Average Basic Hotstuff Latency: {averageBasicLatency:.2f}')
plt.title('Latency of Hotstuff')
plt.xlabel('Latency')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

### EXPERIMENT 6
print("------------ EXPERIMENT 6 ------------")

simulations = 10000
averageWeightedLatency = 0
weightedLatencies = []
for _ in range(simulations):
    Lnew_view = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprepare = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lprecommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)
    Lcommit = generateLatenciesToLeader(n, leaderID=leaderID, low=0, high=1000)

    latency = predictLatencyWeightedHotstuff(n, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit)
    averageWeightedLatency += latency
    weightedLatencies.append(latency)

averageWeightedLatency /= simulations

print(f"The avergae Basic Hotstuff Latency is {averageWeightedLatency}.")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(weightedLatencies, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=averageWeightedLatency, color='red', linestyle='--', label=f'Average Weighted Hotstuff Latency: {averageWeightedLatency:.2f}')
plt.title('Latency of Weighted Hotstuff')
plt.xlabel('Latency')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()