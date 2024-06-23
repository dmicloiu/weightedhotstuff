import numpy as np

from experimental_utils import *
from weighted_hotstuff import *
import matplotlib.pyplot as plt
import os, csv


def experiment(figures_directory, data_directory, timestamp):
    f = 1  # max num of faulty replicas
    delta = 1  # additional replicas
    n = 3 * f + 1 + delta  # total num of replicas

    simulations = 1000

    continuousLatencies = []
    continuousLatenciesFaulty = []

    bestLatencies = []
    bestLatenciesFaulty = []

    latencyDifference = []

    simulationsNumbers = []
    numberOfViews = 10
    leaderRotation = getLeaderRotation(n, numberOfViews=numberOfViews)

    worse = 0
    same = 0
    for simulationNumber in range(simulations):
        networkTopology = generateNetworkTopology(n, 0, 400)
        Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

        (latencyBest, latencyBestFaulty) = weightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, numberOfViews)
        bestLatencies.append(latencyBest)
        bestLatenciesFaulty.append(latencyBestFaulty)

        (latency, latencyFaulty) = continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation,
                                                              numberOfViews)
        continuousLatencies.append(latency)
        continuousLatenciesFaulty.append(latencyFaulty)

        if(latencyFaulty > latencyBestFaulty):
            worse += 1
        elif(latencyFaulty == latencyBestFaulty):
            same += 1

        simulationsNumbers.append(simulationNumber)
        latencyDifference.append(latencyBestFaulty - latencyFaulty)

        print(f"Simulation {simulationNumber} is completed")

    print(f"Continuous Weighted Hotstuff is the same in {same} simulations and better in {1000 - worse - same}.")
    print(f"Best Weighted Hotstuff is better than Continunous one in {worse / simulations * 100:.2f}% of the simulations.")

    plt.figure(figsize=(10, 8))
    plt.plot(simulationsNumbers, latencyDifference, color='blue', marker='o', linestyle='-', linewidth=2,
             markersize=4)

    # add line for average
    plt.axhline(y=np.mean(latencyDifference), color='red', linestyle='--', linewidth=2,
                label=f'Average difference = {np.mean(latencyDifference):.2f} ms')

    plt.xlabel('Simulation number', fontsize=16)
    plt.ylabel('Latency difference [ms]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_directory, f"continuous_{timestamp}.pdf"), bbox_inches='tight')

    # make a bar chart with the results
    results = {
        "worse": worse,
        "same": same,
        "better": 1000 - worse - same,
    }

    # compute corresponding percentages
    total_simulations = 1000
    percentages = {k: v / total_simulations * 100 for k, v in results.items()}

    simulation_types = list(results.keys())
    simulation_results = list(results.values())
    simulation_percentages = list(percentages.values())

    plt.figure(figsize=(10, 8))
    bars = plt.barh(simulation_types, simulation_results, color=['red', 'orange', 'green'])
    plt.xlabel('# Simulations', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add values and percentages beside the bars
    for i, (result, percentage) in enumerate(zip(simulation_results, simulation_percentages)):
        plt.text(result + 10, i, f'{result} ({percentage:.1f}%)', va='center', fontsize=14, style="oblique")

    plt.gca().invert_yaxis()  # Invert y-axis to display simulation types from top to bottom
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory, f"continuous_bar_{timestamp}.pdf"), bbox_inches='tight')

    # create csv file with all data
    csv_filename = os.path.join(data_directory, f'continuous_{timestamp}.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Simulation Number', 'Weighted Hotstuff Latency Faulty (ms)', 'Continuous Hotstuff Latency Faulty (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(simulationsNumbers)):
            writer.writerow({
                'Simulation Number': simulationsNumbers[i],
                'Weighted Hotstuff Latency Faulty (ms)': bestLatenciesFaulty[i],
                'Continuous Hotstuff Latency Faulty (ms)': continuousLatenciesFaulty[i],
            })
