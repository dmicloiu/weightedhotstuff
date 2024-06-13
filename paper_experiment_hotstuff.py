from experimental_utils import *
from weighted_hotstuff import *
import matplotlib.pyplot as plt
import os

def experiment(output_directory, timestamp):
    f = 1 # max num of faulty replicas
    delta = 1  # additional replicas
    n = 3 * f + 1 + delta  # total num of replicas
    leaderID = 0

    networkTopology = [[6.46, 357.86, 222.56, 146.94, 285.3],
                       [361.26, 3.02, 197.7, 211.4, 156.33],
                       [228.36, 198.29, 3.94, 79.35, 80.43],
                       [152.98, 210.13, 78.42, 3.51, 147.77],
                       [290.1, 155.81, 79.21, 147.82, 4.17]]


    weights = set_up_weighting_scheme(networkTopology, delta, f)
    # print(awareWeights)

    basicLatency = []
    basicLatencyFaulty = []

    weightedLatency = []
    weightedFallback = []

    bestLatency = []
    bestFallback = []

    continuousLatency = []
    continuousFallback = []

    bestLeaderLatency = []
    bestLeaderLatencyFaulty = []

    bestWeightsAndLeaderLatency = []
    bestWeightsAndLeaderLatencyFaulty = []

    viewNumbers = []
    for i in range(5, 21):
        viewNumbers.append(i)

    for numberOfViews in viewNumbers:
        leaderRotation = getLeaderRotation(n, numberOfViews)
        Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

        # run in BASIC MODE
        latency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic",
                                      numberOfViews=numberOfViews)
        latency /= numberOfViews
        print("basic: {}".format(latency))
        basicLatency.append(latency)

        latency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic", faulty=True,
                                      numberOfViews=numberOfViews)
        latency /= numberOfViews
        print("basic fallback: {}".format(latency))
        basicLatencyFaulty.append(latency)

        # run in WEIGHTED MODE

        latency = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, weights, leaderRotation, type="weighted",
                                      numberOfViews=numberOfViews)
        latency /= numberOfViews
        latencyFaulty = runWeightedHotstuff(n, f, delta, networkTopology, Lphases, weights, leaderRotation, type="weighted", faulty=True,
                                            numberOfViews=numberOfViews)
        latencyFaulty /= numberOfViews
        weightedLatency.append(latency)
        print("weighted: {}".format(latency))
        print("weighted fallback: {}".format(latencyFaulty))
        weightedFallback.append(latencyFaulty)

        # run in BEST MODE
        (latency, latencyFaulty) = weightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, numberOfViews)
        latency /= numberOfViews
        print("best: {}".format(latency))
        latencyFaulty /= numberOfViews
        print("best faulty: {}".format(latencyFaulty))
        bestLatency.append(latency)
        bestFallback.append(latencyFaulty)

        # run in CONTINUOUS MODE
        (latency, latencyFaulty) = continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, numberOfViews)
        latency /= numberOfViews
        latencyFaulty /= numberOfViews
        continuousLatency.append(latency)
        print("continuous: {}".format(latency))
        continuousFallback.append(latencyFaulty)
        print("continuous faulty: {}".format(latencyFaulty))

        # run in WEIGHTED MODE with LEADER OPTIMALITY
        latency = weightedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, faulty=False)
        latency /= numberOfViews
        print("optimalLeader: {}".format(latency))
        bestLeaderLatency.append(latency)

        latency = weightedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, faulty=True)
        latency /= numberOfViews
        print("optimalLeader fallback: {}".format(latency))
        bestLeaderLatencyFaulty.append(latency)

        # run in BEST MODE with LEADER OPTIMALITY
        latency = weightedBestHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, faulty=False)
        latency /= numberOfViews
        print("best optimalLeader: {}".format(latency))
        bestWeightsAndLeaderLatency.append(latency)

        latency = weightedBestHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, faulty=True)
        latency /= numberOfViews
        print("best optimalLeader fallback: {}".format(latency))
        bestWeightsAndLeaderLatencyFaulty.append(latency)

    # Plot the analysis on different types of behaviours of Hotstuff
    plt.figure(figsize=(14, 10))
    plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=4, markersize=8,
             label='Basic')
    plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=4, markersize=8,
             label='Weighted')

    plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=4, markersize=8,
             label='Best Weighted')

    plt.plot(viewNumbers, continuousLatency, color='red', marker='*', linestyle='-.', linewidth=4, markersize=8,
             label='Continuous Weighted')

    plt.plot(viewNumbers, bestLeaderLatency, color='blue', marker='p', linestyle='--', linewidth=4, markersize=8,
             label='Optimal Leader Rotation Weighted')

    plt.plot(viewNumbers, bestWeightsAndLeaderLatency, color='magenta', marker='o', linestyle='--', linewidth=4, markersize=8,
             label='(Optimal Leader Rotation + Best) Weighted')

    # plt.title('Analysis of Average Latency per View in Hotstuff', fontsize=16)
    plt.xlabel('#views', fontsize=20)
    plt.ylabel('Average Latency per View [ms]', fontsize=20)
    plt.ylim(590, 920)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=18, loc='upper center', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_directory, f"{timestamp}.png"))

    # # Plot the analysis on the fallback efficiency on different types of Hotstuff
    plt.figure(figsize=(14, 10))
    plt.plot(viewNumbers, bestFallback, color='green', marker='d', linestyle=':', linewidth=4, markersize=8,
             label='Best Weighted')

    plt.plot(viewNumbers, continuousFallback, color='red', marker='*', linestyle='-.', linewidth=4, markersize=8,
             label='Continuous Weighted')

    plt.plot(viewNumbers, weightedFallback, color='orange', marker='s', linestyle='--', linewidth=4, markersize=8,
             label='Weighted')

    plt.plot(viewNumbers, bestLeaderLatencyFaulty, color='blue', marker='p', linestyle='--', linewidth=4, markersize=8,
             label='Optimal Leader Rotation Weighted')

    plt.plot(viewNumbers, bestWeightsAndLeaderLatencyFaulty, color='magenta', marker='o', linestyle='--', linewidth=4, markersize=8,
             label='(Optimal Leader Rotation + Best) Weighted')

    # plt.title('Analysis of Fallback Latency Delay in Hotstuff', fontsize=16)
    plt.xlabel('#views', fontsize=20)
    plt.ylabel('Average Latency per View [ms]', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=18, loc='lower center', ncol=2)
    plt.ylim(520)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_directory, f"faulty_{timestamp}.png"))

    # Calculate mean values
    mean_basicLatency = np.mean(basicLatency)
    mean_weightedLatency = np.mean(weightedLatency)
    mean_bestLeaderLatency = np.mean(bestLeaderLatency)
    mean_bestLatency = np.mean(bestLatency)
    mean_continuousLatency = np.mean(continuousLatency)
    mean_bestWeightsAndLeaderLatency = np.mean(bestWeightsAndLeaderLatency)

    # Calculate percentage improvements
    def calculate_improvement(basic, new):
        return ((new - basic) / basic) * 100

    improvements = {
        'Weighted (Non-faulty)': calculate_improvement(mean_basicLatency, mean_weightedLatency),
        'Optimal Leader Weighted': calculate_improvement(mean_basicLatency, mean_bestLeaderLatency),
        'Best': calculate_improvement(mean_basicLatency, mean_bestLatency),
        'Continuous': calculate_improvement(mean_basicLatency, mean_continuousLatency),
        'Optimal Leader Best': calculate_improvement(mean_basicLatency, mean_bestWeightsAndLeaderLatency),
    }

    # Generate LaTeX table code
    latex_table = r"""
    \begin{table}[h]
        \centering
        \caption{Percentage Improvement in Latency Compared to Basic Latency}
        \label{tab:latency_improvement}
        \begin{tabular}{|c|c|c|c|c|c|}
            \hline
            & \textbf{Weighted (Non-faulty)} & \textbf{Optimal Leader Weighted} & \textbf{Best} & \textbf{Continuous} & \textbf{Optimal Leader Best} \\
            \hline
    """

    # Assuming each row corresponds to different figures like in your provided example
    for i, (key, value) in enumerate(improvements.items(), start=1):
        latex_table += f"        \\textbf{{Fig. {i}}} & {value:.2f}\\% \\\\ \n"
        latex_table += "        \hline\n"

    latex_table += r"""
        \end{tabular}
    \end{table}
    """

    print(latex_table)

    # print(np.mean(basicLatency))
    # print(np.mean(weightedLatency))
    # print(np.mean(bestLeaderLatency))
    # print(np.mean(bestLatency))
    # print(np.mean(continuousLatency))
    # print(np.mean(bestWeightsAndLeaderLatency))