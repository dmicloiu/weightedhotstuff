import heapq
import os, csv
import math
import matplotlib.pyplot as plt

from experimental_utils import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def formQuorumWeighted(n, LMessageReceived, weights, quorumWeight):
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


def predictLatencyWeightedHotstuff(n, delta, f, weights, Lnew_view, Lprepare, Lprecommit, Lcommit):
    quorumWeight = 2 * (f + delta) + 1  # quorum formation condition

    # PREPARE phase -> leader waits for quorum formation with (n - f) NEW-VIEW messages from replicas
    tPREPARE = formQuorumWeighted(n, Lnew_view, weights, quorumWeight)

    # after quorum formation -> leader sends PREPARE messages

    # PRE-COMMIT phase -> leader waits for quorum formation with (n - f) PREPARE messages from replicas
    tPRECOMIT = formQuorumWeighted(n, Lprepare, weights, quorumWeight)

    # after quorum formation -> leader sends PRE-COMMIT messages

    # COMMIT phase -> leader waits for quorum formation with (n - f) PRE-COMMIT messages from replicas
    tCOMMIT = formQuorumWeighted(n, Lprecommit, weights, quorumWeight)

    # DECIDE phase -> leader waits for quorum formation with (n - f) COMMIT messages from replicas
    tDECIDE = formQuorumWeighted(n, Lcommit, weights, quorumWeight)

    ## total time of a Hotstuff view run
    return (tPREPARE + tPRECOMIT + tCOMMIT + tDECIDE)


def formQuorumBasic(LMessageReceived, quorumWeight):
    # sort based on the messages that arrive fastest
    LMessageReceived = sorted(LMessageReceived)

    weight = 0
    replicaIdx = 0
    agreementTime = 0
    while weight < quorumWeight:
        agreementTime = LMessageReceived[replicaIdx]
        weight += 1  ## in Basic Hotstuff all relicas have the same weight
        replicaIdx += 1

    return agreementTime


def predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit):
    quorumWeight = np.ceil((n + f + 1) / 2)  # quorum formation condition -> majority for egalitarian quorum

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

    # print('--------------------------------')
    # print('--------' + suffix + ' Simulated annealing')
    # print('--------------------------------')
    # print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    # print('Final solution latency:', bestLat)
    # print('Best Configuration:  R_max: {}, weight: {} | R_min: {}, weight: {} with leader {}'.format(r_max, vmax, r_min,
    #                                                                                                  vmin, bestLeader))
    # print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    # print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))

    return bestLat

def generateAllPossibleLeaderRotations(n, numberOfViews):
    leaderRotations = []
    for i in range(n):
        leaderRotations.append([i])

    for i in range(1, numberOfViews):
        newLeaderRotations = []
        for leaderRotation in leaderRotations:

            for replicaID in range(n):
                if replicaID not in leaderRotation:
                    newLeaderRotations.append(leaderRotation + [replicaID])

        leaderRotations = newLeaderRotations

    return leaderRotations


def experiment(figures_directory, data_directory, timestamp):
    f = 1  # max num of faulty replicas
    delta = 1  # additional replicas
    n = 3 * f + 1 + delta  # total num of replicas

    vmax = 1 + delta / f  # 2f replicas
    vmin = 1  # n - 2f replicas

    weights = []
    for i in range(2 * f):
        weights.append(vmax)
    for i in range(n - 2 * f):
        weights.append(vmin)


    networkTopology = [[9.74, 290.12, 222.89, 149.97, 284.75],
                        [286.24, 2.21, 202.91, 210.16, 156.58],
                        [226.65, 203.8, 6.01, 81.86, 80.49],
                        [153.52, 210.18, 79.96, 5.1, 148.79],
                        [288.76, 155.68, 79.87, 148.54, 3.97]]

    numberOfViews = 4

    possibleLeaderRotations = generateAllPossibleLeaderRotations(n, numberOfViews)

    latencies = []
    for leaderRotation in possibleLeaderRotations:
        Lphases = generateExperimentLatencies(n, numberOfViews, networkTopology, leaderRotation)

        latency = 0
        for viewNumber in range(numberOfViews):
            (Lnew_view, Lprepare, Lprecommit, Lcommit) = Lphases[viewNumber]
            latency += predictLatencyBasicHotstuff(n, f, Lnew_view, Lprepare, Lprecommit, Lcommit)
        latencies.append(latency)

    # encode each unique rotation as a categorical variable
    rotation_strings = ['-'.join(map(str, rotation)) for rotation in possibleLeaderRotations]

    # create DataFrames
    df_rotations = pd.DataFrame(rotation_strings, columns=["rotation"])
    df_latencies = pd.DataFrame(latencies, columns=["latency"])

    # DEBUG purposes
    # print(df_rotations)
    # print(df_latencies)

    # combine the DataFrames
    df = pd.concat([df_rotations, df_latencies], axis=1)

    # encode the rotation strings to numerical values
    encoder = LabelEncoder()
    encoded_rotations = encoder.fit_transform(df['rotation'])

    # add encoded rotations to the DataFrame
    df['encoded_rotation'] = encoded_rotations

    # sort DataFrame by encoded_rotation for better visualization
    df_sorted = df.sort_values(by='encoded_rotation')

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(df_sorted['encoded_rotation'], df_sorted['latency'], c=df_sorted['latency'], cmap='coolwarm', edgecolor='k', s=100)

    # add the gradual colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Latency [ms]', fontsize=18)
    cbar.ax.yaxis.set_tick_params(labelsize=16)

    # plt.title('Analysis of Leader Rotation impact on Latency in Hotstuff', fontsize=18)
    plt.xlabel('Encoded Leader Rotation', fontsize=18)
    plt.ylabel('Latency [ms]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_directory, f"leader_rotation_{timestamp}.pdf"), bbox_inches='tight')

    # create csv file with all data
    csv_filename = os.path.join(data_directory, f'leader_rotation_{timestamp}.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Simulation', 'Leader Rotation', 'Latency (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(possibleLeaderRotations)):
            writer.writerow({
                'Simulation': i,
                'Leader Rotation': possibleLeaderRotations[i],
                'Latency (ms)': latencies[i],
            })

