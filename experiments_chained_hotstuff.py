import matplotlib.pyplot as plt
import numpy as np

from experimental_utils import *
from chained_weighted_hotstuff import *

f = 1  # max num of faulty replicas
delta = 1  # additional replicas
n = 3 * f + 1 + delta  # total num of replicas

numberOfViews = 10 # total number of views in a Hotstuff simulation

networkTopology = generateNetworkTopology(n, 0, 400)

awareWeights = [1] * numberOfViews
for i in range(2 * f):
    awareWeights[i] = 1 + delta / f


print(f"All following experiments are run with n = {n}, number of views = {numberOfViews}")

## EXPERIMENT 1
print("------------ EXPERIMENT 1 ------------")
simulations = 10000

avgBasicLatency = 0
avgWeightedLatency = 0

basicLatency = []
weightedLatency = []


for _ in range(simulations):
    leaderRotation = getLeaderRotation(n, numberOfViews)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

    # run in BASIC MODE
    latency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='basic', numberOfViews=numberOfViews, faulty=False)
    basicLatency.append(latency)
    avgBasicLatency += (latency / simulations)

    # run in WEIGHTED MODE
    latency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=False)
    weightedLatency.append(latency)
    avgWeightedLatency += (latency / simulations)


print(f"We perform Chained Hotstuff using {simulations} simulations of the protocol, using {numberOfViews} views.")
print(f"Average latency of Basic Chained Hotstuff: {avgBasicLatency}")
print(f"Average latency of Weighted Chained Hotstuff: {avgWeightedLatency}")

# Graphical representation for BASIC MODE
plt.figure(figsize=(8, 6))
plt.hist(basicLatency, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=avgBasicLatency, color='red', linestyle='--', label=f'Average Basic Latency: {avgBasicLatency:.2f}')
plt.title('Latency of Chained Hotstuff')
plt.xlabel('Latency [ms]')
plt.ylabel('Number of Simulations')
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
plt.legend()
plt.grid(True)
plt.show()


# EXPERIMENT 2 -> one simulation over multiple view numbers to see potential trends
basicLatency = []
weightedLatency = []
bestLatency = []
bestLeaderLatency = []

viewNumbers = []
for i in range(1, 10):
    viewNumbers.append(i * 5)

for numberOfViews in viewNumbers:
    leaderRotation = getLeaderRotation(n, numberOfViews)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

    # run in BASIC MODE
    latency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='basic', numberOfViews=numberOfViews, faulty=False) / numberOfViews
    basicLatency.append(latency)

    # run in WEIGHTED MODE
    latency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=False) / numberOfViews
    weightedLatency.append(latency)

    # run in BEST MODE
    latency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='best', numberOfViews=numberOfViews, faulty=False) / numberOfViews
    bestLatency.append(latency)

    # run in WEIGHTED MODE with leader rotation optimisation
    latency = chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, awareWeights, numberOfViews, type='weighted', faulty=False) / numberOfViews
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


## EXPERIMENT 3
print("------------ EXPERIMENT 3 ------------")

numberOfViews = 10

basicLatency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='basic', numberOfViews=numberOfViews, faulty=False)
weightedLatency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=False)
weightedLatencyFaulty = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=True)
bestLatency = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='best', numberOfViews=numberOfViews, faulty=False)
bestLatencyFaulty = setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation,
                                                       type='best', numberOfViews=numberOfViews, faulty=True)
weightedLatencyBestLeader = chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, awareWeights, numberOfViews, type='weighted', faulty=False)
bestLatencyBestLeader = chainedHotstuffBestAndOptimalLeader(n, f, delta, networkTopology, Lphases, numberOfViews)

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
    "Non-faulty": round(weightedLatency, 2),
    "Faulty": round(weightedLatencyFaulty, 2),
    "Best": round(bestLatency, 2),
    "Best Faulty": round(bestLatencyFaulty, 2),
    "Optimal Leader": round(weightedLatencyBestLeader, 2),
    "Optimal Leader Best": round(bestLatencyBestLeader, 2),
}

simulation_types = list(results.keys())
simulation_results = list(results.values())

plt.figure(figsize=(12, 10))
bars = plt.barh(simulation_types, simulation_results, color=['skyblue', 'red', 'green', 'red', 'skyblue', 'green'])
plt.xlabel('Latency [ms]', fontsize=14)
# plt.ylabel('Weighting Assignment', fontsize=14)
# plt.title(f'Weighted Chained Hotstuff performance over {numberOfViews} views', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding values beside the bars
for i, result in enumerate(simulation_results):
    plt.text(result + 0.1, i, str(result), va='center', fontsize=10, style="oblique")

plt.axvline(basicLatency, color='darkred', linestyle='--', linewidth=2, label=f'Chained Hotstuff (Baseline) = {basicLatency:.2f} ms')
plt.title("Weighted Chained Hotstuff performance over 10 views")
plt.gca().invert_yaxis()  # invert y-axis to display simulation types from top to bottom
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.legend(loc='upper right', fontsize=12)
plt.show()
