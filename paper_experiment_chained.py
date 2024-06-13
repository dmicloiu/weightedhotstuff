from chained_weighted_hotstuff import *
import matplotlib.pyplot as plt
import os, csv


def experiment(figures_directory, data_directory, timestamp):
    f = 1  # max num of faulty replicas
    delta = 1  # additional replicas
    n = 3 * f + 1 + delta  # total num of replicas

    networkTopology = [[6.46, 357.86, 222.56, 146.94, 285.3],
                       [361.26, 3.02, 197.7, 211.4, 156.33],
                       [228.36, 198.29, 3.94, 79.35, 80.43],
                       [152.98, 210.13, 78.42, 3.51, 147.77],
                       [290.1, 155.81, 79.21, 147.82, 4.17]]
    weights = set_up_weighting_scheme(networkTopology, delta, f)

    basicLatency = []
    weightedLatency = []
    weightedLatencyFaulty = []
    bestLatency = []
    bestLatencyFaulty = []
    weightedLatencyBestLeader = []
    weightedLatencyBestLeaderFaulty = []
    bestLatencyBestLeader = []

    viewNumbers = []
    for i in range(5, 21):
        viewNumbers.append(i)

    for numberOfViews in viewNumbers:
        leaderRotation = getLeaderRotation(n, numberOfViews)
        Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

        basicLatency.append(
            setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='basic', numberOfViews=numberOfViews, faulty=False) / numberOfViews)

        weightedLatency.append(
            setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='weighted', numberOfViews=numberOfViews, faulty=False) / numberOfViews)
        weightedLatencyFaulty.append(
            setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='weighted', numberOfViews=numberOfViews, faulty=True) / numberOfViews)

        bestLatency.append(
            setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='best', numberOfViews=numberOfViews, faulty=False) / numberOfViews)
        bestLatencyFaulty.append(
            setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='best', numberOfViews=numberOfViews, faulty=True) / numberOfViews)

        weightedLatencyBestLeader.append(
            chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, type='weighted',
                                         faulty=False) / numberOfViews)
        weightedLatencyBestLeaderFaulty.append(
            chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, type='weighted',
                                         faulty=True) / numberOfViews)

        bestLatencyBestLeader.append(
            chainedHotstuffBestAndOptimalLeader(n, f, delta, networkTopology, Lphases, numberOfViews) / numberOfViews)

        # uncomment line below to visualise when a specific simulation for a number of views is finished
        # print(numberOfViews)

    plt.figure(figsize=(14, 10))
    plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=4, markersize=8,
             label='Basic')
    plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=4, markersize=8,
             label='Weighted')
    # plt.plot(viewNumbers, weightedLatencyFaulty, color='darkred', marker='s', linestyle='--', linewidth=2, markersize=6,
    #          label='Faulty')
    plt.plot(viewNumbers, weightedLatencyBestLeader, color='blue', marker='*', linestyle='-.', linewidth=4,
             markersize=8,
             label='Optimal Leader Rotation Weighted')
    plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=4, markersize=8,
             label='Best Weighted')
    # plt.plot(viewNumbers, bestLatencyFaulty, color='black', marker='d', linestyle=':', linewidth=2, markersize=6,
    #          label='Best Faulty')
    plt.plot(viewNumbers, bestLatencyBestLeader, color='magenta', marker='D', linestyle='--', linewidth=4, markersize=8,
             label='(Optimal Leader Rotation + Best) Weighted')

    # plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
    plt.xlabel('#views', fontsize=20)
    plt.ylabel('Average Latency per View [ms]', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_directory, f"chained_{timestamp}.png"))

    plt.figure(figsize=(14, 10))
    plt.plot(viewNumbers, weightedLatencyFaulty, color='orange', marker='s', linestyle='--', linewidth=4, markersize=8,
             label='Weighted')

    plt.plot(viewNumbers, weightedLatencyBestLeaderFaulty, color='blue', marker='*', linestyle='-.', linewidth=4,
             markersize=8,
             label='Optimal Leader Rotation Weighted')

    plt.plot(viewNumbers, bestLatencyFaulty, color='green', marker='d', linestyle=':', linewidth=4, markersize=8,
             label='Best Weighted')

    # plt.title('Analysis of Average Latency per View i Chained Hotstuff', fontsize=16)
    plt.xlabel('#views', fontsize=20)
    plt.ylabel('Average Latency per View [ms]', fontsize=20)
    plt.legend(fontsize=17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_directory, f"chained_faulty_{timestamp}.png"))

    #  calculate mean values
    mean_basicLatency = np.mean(basicLatency)
    mean_weightedLatency = np.mean(weightedLatency)
    mean_bestLeaderLatency = np.mean(weightedLatencyBestLeader)
    mean_bestLatency = np.mean(bestLatency)
    mean_bestWeightsAndLeaderLatency = np.mean(bestLatencyBestLeader)

    # calculate percentage improvements
    improvements = {
        'Weighted (Non-faulty)': calculate_improvement(mean_basicLatency, mean_weightedLatency),
        'Optimal Leader Weighted': calculate_improvement(mean_basicLatency, mean_bestLeaderLatency),
        'Best': calculate_improvement(mean_basicLatency, mean_bestLatency),
        'Optimal Leader Best': calculate_improvement(mean_basicLatency, mean_bestWeightsAndLeaderLatency),
    }

    # create csv file with all data
    csv_filename = os.path.join(data_directory, f'chained_{timestamp}.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Protocol', 'Mean Latency (ms)', 'Improvement (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Protocol': 'Basic', 'Mean Latency (ms)': mean_basicLatency, 'Improvement (%)': 0})
        writer.writerow({'Protocol': 'Weighted (Non-faulty)', 'Mean Latency (ms)': mean_weightedLatency,
                         'Improvement (%)': improvements['Weighted (Non-faulty)']})
        writer.writerow({'Protocol': 'Optimal Leader Weighted', 'Mean Latency (ms)': mean_bestLeaderLatency,
                         'Improvement (%)': improvements['Optimal Leader Weighted']})
        writer.writerow({'Protocol': 'Best', 'Mean Latency (ms)': mean_bestLatency, 'Improvement (%)': improvements['Best']})
        writer.writerow({'Protocol': 'Optimal Leader Best', 'Mean Latency (ms)': mean_bestWeightsAndLeaderLatency,
                         'Improvement (%)': improvements['Optimal Leader Best']})


