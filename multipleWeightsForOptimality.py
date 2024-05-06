import random
import heapq
from collections import Counter
from itertools import chain, repeat, islice, count, combinations
import string
import scipy.special
import time
import matrix_utils as m_utils
##import triplet_utils as t_utils
import math


def formQV(n, Mwrite, tProposed, weights, quorumWeight):  # Alg. 1 in AWARE

    received = []
    for i in range(n):
        received.append([])
        for j in range(n):
            heapq.heappush(received[i], (tProposed[j] + Mwrite[j][i], weights[j]))  # use received[] as a priority queue

    nextTimes = [0] * n

    quorumTimes = []
    for i in range(n):
        weight = 0
        while weight < quorumWeight:
            (rcvTime, rcvWeight) = heapq.heappop(received[i])
            weight += rcvWeight
            nextTimes[i] = rcvTime

    return nextTimes


def predictLatency(n, f, delta, weights, leaderId, Mpropose, Mwrite, rounds):  # Alg. 2 in AWARE

    quorumWeight = 2 * (f + delta) + 1  # Weight of a quorum

    tProposed = [0] * n  # time at which the latest proposal is received by replicas
    offset = [0] * n  # time at which replicas deliver the latest proposal

    tAccepted = [0] * n

    consensusLatencies = [0] * rounds  # latency of each consensus round

    for r in range(rounds):

        # compute times at which each replica receives the leader's i-th propose
        for i in range(n):
            if i != leaderId:  # not in Alg. 2
                tProposed[i] = max(tProposed[leaderId] + Mpropose[leaderId][i],
                                   tAccepted[i])  # added tProposed[leaderId]
        print('sending WRITE times', tProposed)

        tWritten = formQV(n, Mwrite, tProposed, weights, quorumWeight)
        print('sending ACCEPT times', tWritten)

        tAccepted = formQV(n, Mwrite, tWritten, weights, quorumWeight)
        print('ACCEPTED times', tAccepted, '\n')

        consensusLatencies[r] = tAccepted[leaderId] - tProposed[leaderId]
        tProposed[leaderId] = tAccepted[leaderId]  # not in Alg. 2

    print(consensusLatencies)
    return sum(consensusLatencies) / len(consensusLatencies)


def convert_bestweights_to_rmax_rmin(best_weights):
    replicas = [i for i in range(len(best_weights))]
    r_max = []
    r_min = []
    for rep_id in replicas:
        ## !!! VERIFY IF THE WEIGHTING SCHEME PERMITS THE CUSTOM WEIGHTS TO BE 1
        if best_weights[rep_id] != vmin:
            r_max.append((replicas[rep_id], best_weights[rep_id]))
        else:
            r_min.append((replicas[rep_id], best_weights[rep_id]))
    return r_max, r_min


def simulated_annealing(n, f, delta, Mpropose, Mwrite, r, suffix):
    start = time.time()

    random.seed(500)

    step = 0
    step_max = 1000000
    temp = 120
    init_temp = temp
    theta = 0.0055
    t_min = 0.2
    r_max = []
    r_min = []

    ## SHIFT IN 1 - N
    curWeights = [1] * n
    for i in range(1, 2 * f + 1):
        curWeights[i - 1] = 1 + i * delta / m
    curLeader = 0

    curLat = predictLatency(n, f, delta, curWeights, curLeader, Mpropose, Mwrite, r)

    bestLat = curLat
    bestLeader = -1
    bestWeights = []
    jumps = 0

    while step < step_max and temp > t_min:
        replicaFrom = -1
        replicaTo = -1
        newLeader = curLeader
        while True:
            replicaFrom = random.randint(0, n - 1)
            if curWeights[replicaFrom] != vmin:
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
        ##    print(newWeights)

        newLat = predictLatency(n, f, delta, newWeights, newLeader, Mpropose, Mwrite, r)

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
    r_max, r_min = convert_bestweights_to_rmax_rmin(bestWeights)

    print('--------------------------------')
    print('--------' + suffix + ' Simulated annealing')
    print('--------------------------------')
    print('Configurations examined: {}    time needed:{}'.format(step, end - start))
    print('Final solution latency:', bestLat)
    print('Best Configuration:  R_max: {}  | R_min: {}  with leader {}'.format(r_max, r_min, bestLeader))
    print('initTemp:{} finalTemp:{}'.format(init_temp, temp))
    print('coolingRate:{} threshold:{} jumps:{}'.format(theta, t_min, jumps))


### CONFIGURABLE SIDE
f = 1  # max num of faulty replicas
delta = 1  # additional replicas

n = 3 * f + 1 + delta  # total num of replicas

m = f * (f + 1) / 2  # weighting coefficient

## Vmin is same as V0 in the support document
vmin = 1  # n - 2f replicas will still have minimal weight 1 -> to ensure avilability of quorum formation

weights = []
## change to start from 1 so that the weighting scheme is appropriate
for i in range(1, 2 * f + 1):
    weight_i = 1 + i * delta / m
    ### observation -> we assign to the reamining f replicas each a diff weight Vi -> if leader 0, it will take weight delta / m
    # -> hence ask if we should do it in reverse order to give higher probability to the leader
    weights.append(weight_i)

for i in range(n - 2 * f):
    weights.append(vmin)

leaderId = 0
rounds = 10

Mpropose = m_utils.generateRandomMatrix(n, 0, 1000)
Mwrite = m_utils.generateRandomMatrix(n, 0, 1000)

# print(predictLatency(n, f, delta, weights, leaderId, Mpropose, Mwrite, rounds))


suffix=''
simulated_annealing(n,f,delta,Mpropose,Mwrite,rounds,suffix)