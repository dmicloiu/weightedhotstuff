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

    weightedLatencies = []
    weightedLatenciesFaulty = []

    simulationsNumbers = []
    numberOfViews = 10
    leaderRotation = getLeaderRotation(n, numberOfViews=numberOfViews)

    worse = 0

    for simulationNumber in range(simulations):
        networkTopology = generateNetworkTopology(n, 0, 400)
        # set up weighting scheme
        awareWeights = set_up_weighting_scheme(networkTopology, delta, f)
        # print(awareWeights)

        Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

        simulationsNumbers.append(simulationNumber)

        weightedLatencies.append(runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation, type="weighted",
                                      numberOfViews=numberOfViews))
        weightedLatenciesFaulty.append(runWeightedHotstuff(n, f, delta, networkTopology, Lphases, awareWeights, leaderRotation, type="weighted",
                                      numberOfViews=numberOfViews, faulty=True))

        (latency, latencyFaulty) = continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation,
                                                              numberOfViews)
        continuousLatencies.append(latency)
        continuousLatenciesFaulty.append(latencyFaulty)

        if(continuousLatenciesFaulty[-1] > weightedLatenciesFaulty[-1]):
            worse += 1

        print(f"Simulation {simulationNumber} is completed")


    print(f"Weighted Hotstuff is better than Continunous one in {worse / simulations * 100:.2f}% of the simulations.")

    plt.figure(figsize=(10, 8))

    plt.plot(simulationsNumbers, weightedLatenciesFaulty, color='skyblue', marker='o', linestyle='--', linewidth=2, markersize=6,
             label='Weighted')
    plt.plot(simulationsNumbers, continuousLatenciesFaulty, color='purple', marker='d', linestyle=':', linewidth=2, markersize=6,
             label='Continuous')

    # plt.title('Analysis of Average Latency per View in Hotstuff', fontsize=16)
    plt.xlabel('Simulation number', fontsize=16)
    plt.ylabel('Latency [ms]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_directory, f"continuous_{timestamp}.png"))

    # create csv file with all data
    csv_filename = os.path.join(data_directory, f'continuous_{timestamp}.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Simulation Number', 'Weighted Hotstuff Latency Faulty (ms)', 'Continuous Hotstuff Latency Faulty (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(simulationsNumbers)):
            writer.writerow({
                'Simulation Number': simulationsNumbers[i],
                'Weighted Hotstuff Latency Faulty (ms)': weightedLatenciesFaulty[i],
                'Continuous Hotstuff Latency Faulty (ms)': continuousLatenciesFaulty[i],
            })