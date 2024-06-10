from experimental_utils import *
from chained_weighted_hotstuff import *
import matplotlib.pyplot as plt


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
basicLatencyFaulty = []
weightedLatency = []
weightedLatencyFaulty = []
bestLatency = []
bestLatencyFaulty = []
bestLeaderLatency = []
weightedLatencyBestLeader = []
weightedLatencyBestLeaderFaulty = []
bestLatencyBestLeader = []


viewNumbers = []
for i in range(5, 21):
    viewNumbers.append(i)


for numberOfViews in viewNumbers:
    leaderRotation = getLeaderRotation(n, numberOfViews)
    Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

    basicLatency.append(setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                                       type='basic', numberOfViews=numberOfViews, faulty=False) / numberOfViews)

    weightedLatency.append(setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=False) / numberOfViews)
    weightedLatencyFaulty.append(
        setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                                       type='weighted', numberOfViews=numberOfViews, faulty=True) / numberOfViews)

    bestLatency.append(setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                                       type='best', numberOfViews=numberOfViews, faulty=False) / numberOfViews)
    bestLatencyFaulty.append(
        setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                                       type='best', numberOfViews=numberOfViews, faulty=True)/ numberOfViews)

    weightedLatencyBestLeader.append(chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, type='weighted', faulty=False) / numberOfViews)
    weightedLatencyBestLeaderFaulty.append(
        chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, numberOfViews, type='weighted', faulty=True) / numberOfViews)

    bestLatencyBestLeader.append(chainedHotstuffBestAndOptimalLeader(n, f, delta, networkTopology, Lphases, numberOfViews) / numberOfViews)

    # uncomment line below to visualise when a specific simulation for a number of views is finished
    # print(numberOfViews)


plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, basicLatency, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=6,
         label='Basic')
plt.plot(viewNumbers, weightedLatency, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Weighted')
# plt.plot(viewNumbers, weightedLatencyFaulty, color='darkred', marker='s', linestyle='--', linewidth=2, markersize=6,
#          label='Faulty')
plt.plot(viewNumbers, weightedLatencyBestLeader, color='blue', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Optimal Leader Weighted')
plt.plot(viewNumbers, bestLatency, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best Weighted')
# plt.plot(viewNumbers, bestLatencyFaulty, color='black', marker='d', linestyle=':', linewidth=2, markersize=6,
#          label='Best Faulty')
plt.plot(viewNumbers, bestLatencyBestLeader, color='magenta', marker='D', linestyle='--', linewidth=2, markersize=6,
         label='(Optimal Leader + Best) Weighted')

# plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('#views', fontsize=12)
plt.ylabel('Average Latency per View [ms]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10, 8))
plt.plot(viewNumbers, weightedLatencyFaulty, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6,
         label='Weighted')

plt.plot(viewNumbers, weightedLatencyBestLeaderFaulty, color='blue', marker='*', linestyle='-.', linewidth=2, markersize=6,
         label='Optimal Leader Weighted')

plt.plot(viewNumbers, bestLatencyFaulty, color='green', marker='d', linestyle=':', linewidth=2, markersize=6,
         label='Best Weighted')

# plt.title('Analysis of Average Latency per View in Chained Hotstuff', fontsize=16)
plt.xlabel('#views', fontsize=12)
plt.ylabel('Average Latency per View [ms]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# # Calculate mean values
# mean_basicLatency = np.mean(basicLatency)
# mean_weightedLatency = np.mean(weightedLatency)
# mean_bestLeaderLatency = np.mean(weightedLatencyBestLeader)
# mean_bestLatency = np.mean(bestLatency)
# mean_bestWeightsAndLeaderLatency = np.mean(bestLatencyBestLeader)

# print(np.mean(basicLatency))
# print(np.mean(weightedLatency))
# print(np.mean(weightedLatencyBestLeader))
# print(np.mean(bestLatency))
# print(np.mean(bestLatencyBestLeader))

#
# # Calculate percentage improvements
# def calculate_improvement(basic, new):
#     return ((new - basic) / basic) * 100
#
# improvements = {
#     'Weighted (Non-faulty)': calculate_improvement(mean_basicLatency, mean_weightedLatency),
#     'Optimal Leader Weighted': calculate_improvement(mean_basicLatency, mean_bestLeaderLatency),
#     'Best': calculate_improvement(mean_basicLatency, mean_bestLatency),
#     'Optimal Leader Best': calculate_improvement(mean_basicLatency, mean_bestWeightsAndLeaderLatency),
# }
#
# # Generate LaTeX table code
# latex_table = r"""
# \begin{table}[h]
#     \centering
#     \caption{Percentage Improvement in Latency Compared to Basic Latency}
#     \label{tab:latency_improvement}
#     \begin{tabular}{|c|c|c|c|c|c|}
#         \hline
#         & \textbf{Weighted (Non-faulty)} & \textbf{Optimal Leader Weighted} & \textbf{Best} & \textbf{Continuous} & \textbf{Optimal Leader Best} \\
#         \hline
# """
#
# # Assuming each row corresponds to different figures like in your provided example
# for i, (key, value) in enumerate(improvements.items(), start=1):
#     latex_table += f"        \\textbf{{Fig. {i}}} & {value:.2f}\\% \\\\ \n"
#     latex_table += "        \hline\n"
#
# latex_table += r"""
#     \end{tabular}
# \end{table}
# """
#
# print(latex_table)
#