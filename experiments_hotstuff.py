import numpy as np

from experimental_utils import *
from weighted_hotstuff import *
import matplotlib.pyplot as plt

f = 1 # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas
leaderID = 0

## EXPERIMENT 1
print("------------ EXPERIMENT 1 ------------")

leaderRotation = [0]
numberOfViews = 1

# set up weighting scheme
awareWeights = [1] * n
for i in range(2 * f):
    awareWeights[i] = 1 + delta / f

# set up the network scenario
networkTopology = generateNetworkTopology(n, 0, 400)
Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)
(Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]

basicLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic",
                                  numberOfViews=numberOfViews)

weightedLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation, type="weighted",
                                  numberOfViews=numberOfViews)

(bestLatency, bestLatencyFaulty) = weightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, numberOfViews)

print(f"Basic Hotstuff with leader {leaderID} completes a view in {basicLatency}.")
print(f"Weighted Hotstuff with leader {leaderID} completes a view in {weightedLatency}.")
print(f"Best Weighted Hotstuff with leader {leaderID} completes a view in {bestLatency}.")
print("\n")

### EXPERIMENT 2
print("------------ EXPERIMENT 2 ------------")

# experiment setup
simulations = 10000
leaderRotation = [0]
numberOfViews = 1

# set up weighting scheme
awareWeights = [1] * n
for i in range(2 * f):
    awareWeights[i] = 1 + delta / f

averageDifference = 0
timesBasicIsFaster = 0
timesEqualPerformance = 0

differences = []

for _ in range(simulations):
    networkTopology = generateNetworkTopology(n, 0, 400)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)
    (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]

    # take into consideration the randomness of weighted Hotstuff
    np.random.shuffle(awareWeights)

    basicLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic",
                                  numberOfViews=numberOfViews)
    weightedLatency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation, type="weighted",
                                  numberOfViews=numberOfViews)

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
print(f"The two algorithms have equal performance in {timesEqualPerformance} simulations.")
print(f"Weighted Hotstuff is on average with {averageDifference} faster than the Basic version.")
print("\n")

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

simulations = 10000
leaderRotation = [0]

averageBasicLatency = 0
basicLatencies = []
for _ in range(simulations):
    networkTopology = generateNetworkTopology(n, 0, 400)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)
    (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]

    latency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic",
                                  numberOfViews=numberOfViews)
    averageBasicLatency += latency
    basicLatencies.append(latency)

averageBasicLatency /= simulations

print(f"The average Basic Hotstuff Latency is {averageBasicLatency}.\n")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(basicLatencies, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=averageBasicLatency, color='red', linestyle='--', label=f'Average Basic Hotstuff Latency: {averageBasicLatency:.2f}')
plt.title('Latency of Basic Hotstuff')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

### EXPERIMENT 4
print("------------ EXPERIMENT 4 ------------")

simulations = 10000
leaderRotation = [0]

# set up weighting scheme
awareWeights = [1] * n
for i in range(2 * f):
    awareWeights[i] = 1 + delta / f

averageWeightedLatency = 0
weightedLatencies = []
for _ in range(simulations):
    networkTopology = generateNetworkTopology(n, 0, 400)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)
    (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]

    latency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation, type="weighted",
                                  numberOfViews=numberOfViews)

    averageWeightedLatency += latency
    weightedLatencies.append(latency)

averageWeightedLatency /= simulations

print(f"The average Weighted Hotstuff Latency is {averageWeightedLatency}.")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(weightedLatencies, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=averageWeightedLatency, color='red', linestyle='--', label=f'Average Weighted Hotstuff Latency: {averageWeightedLatency:.2f}')
plt.title('Latency of Weighted Hotstuff')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()

### EXPERIMENT 5
print("------------ EXPERIMENT 5 ------------")

f_values = [1, 2, 3, 4]
continuous_weighted_hotstuff_performance = []
numberOfViews = 1

for f in f_values:
    delta = 1  # additional replicas
    n = 3 * f + 1 + delta  # total num of replicas

    # print(f)

    leaderRotation = getLeaderRotation(n, numberOfViews)

    awareWeights = [1] * n
    for i in range(2 * f):
        awareWeights[i] = 1 + delta / f

    # set up the network scenario
    networkTopology = generateNetworkTopology(n, 0, 400)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)
    (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[0]

    start = time.time()
    continuous_weighted_hotstuff_performance.append(continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, numberOfViews))
    end = time.time()

    print(f"Running a continunous Hotstuff simulation with {numberOfViews} views and {n} nodes took {end - start}")
