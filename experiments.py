import argparse
import os, datetime, csv

import paper_experiment_continuous
import paper_leader_rotation
import paper_experiment_chained
import paper_experiment_hotstuff
from chained_weighted_hotstuff import *
from weighted_hotstuff import *

# create directories for result figures and data gathered in CSV files
def create_directory(base_directory="results"):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    figures_dir = os.path.join(base_directory, "figures")
    data_dir = os.path.join(base_directory, "data")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return (figures_dir, data_dir)

# options for running the experiments
parser = argparse.ArgumentParser(description='Experiment simulations for Hotstuff')
parser.add_argument("--chained", action="store_true", default=False, help="Run experiments with Chained Hotstuff.")
parser.add_argument("--paper", action="store_true", default=False, help="Run experiments from paper.")
parser.add_argument("--sim", type=int, default=1, help="Number of simulations.")
parser.add_argument("--views-lower-bound", type=int, default=1, help="The lower bound of the number of views in a protocol run.")
parser.add_argument("--views-upper-bound", type=int, default=1, help="The upper bound of the number of views in a protocol run.")
parser.add_argument("--network-setup", type=int, default=0, help="Run experiment using the following network setup.")
parser.add_argument("--f", type=int, default=1, help="Number of failures the system can withstand.")
parser.add_argument("--delta", type=int, default=1, help="Number of additional replicas.")
parser.add_argument("--faulty", action="store_true", default=False, help="Run experiments in faulty scenario.")
parser.add_argument("--basic", action="store_true", default=False, help="Run experiments unweighted.")
parser.add_argument("--weighted", action="store_true", default=False, help="Run experiments weighted.")
parser.add_argument("--best", action="store_true", default=False, help="Run experiments for optimising weight distribution.")
parser.add_argument("--lr", action="store_true", default=False, help="Run experiments for optimising leader rotation.")
parser.add_argument("--continuous", action="store_true", default=False, help="Run experiment with continuous weighted scheme.")
parser.add_argument("--all", action="store_true", default=False, help="Run experiment with all variants of the protocol.")
args = parser.parse_args()


figures_directory, data_directory = create_directory()
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

f = args.f
delta = args.delta
n = 3 * f + 1 + delta

# edge case for wrong use of flags
if args.lr and args.views_lower_bound < n:
    args.views_lower_bound = n

if args.views_lower_bound > args.views_upper_bound:
    args.views_upper_bound = args.views_lower_bound


simulations = [i for i in range(args.sim)]
viewNumbers = [i for i in range(args.views_lower_bound, args.views_upper_bound + 1)]

if(args.network_setup == 1):
    # run experiment on the 5 nodes network topology used in the paper
    networkTopology = paper_networkTopology

    # update parameters accordingly
    f = 1
    delta = 1
    n = 3 * f + 1 + delta

elif(args.network_setup == 2):
    # run experiment on 8 nodes network topology with cloudping data
    networkTopology = two_faults_networkTopology

    # update parameters accordingly
    f = 2
    delta = 1
    n = 3 * f + 1 + delta

elif(args.network_setup == 3):
    # run experiment on 11 nodes network topology with cloudping data
    networkTopology = three_faults_networkTopology

    # update parameters accordingly
    f = 3
    delta = 1
    n = 3 * f + 1 + delta

else:
    networkTopology = generateNetworkTopology(n, 0, 400)

weights = set_up_weighting_scheme(networkTopology, delta, f)

if args.all:
    args.basic = True
    args.weighted = True
    args.best = True
    args.continuous = True
    args.lr = True

if args.paper:
    # run paper Chained Hotstuff experiment
    if args.chained:
        paper_experiment_chained.experiment(figures_directory, data_directory, timestamp)

    # run paper Leader Rotation impact analysis
    elif args.lr:
        paper_leader_rotation.experiment(figures_directory, data_directory, timestamp)

    # run paper Continuous Hotstuff experiment
    elif args.continuous:
        paper_experiment_continuous.experiment(figures_directory, data_directory, timestamp)

    # run paper Hotstuff experiment
    else:
        paper_experiment_hotstuff.experiment(figures_directory, data_directory, timestamp)

# run other experiments on CHAINED HOTSTUFF
elif args.chained:
    # gather all data
    csv_filename = os.path.join(data_directory, f'experiment_{timestamp}.csv')

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldNames = ['Simulation Number', 'View Number', 'Type', 'Faulty', 'Latency (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)

        writer.writeheader()

        for simulation_number in range(args.sim):
            for views_number in viewNumbers:
                leaderRotation = getLeaderRotation(n, views_number)
                Lphases = generateExperimentLatencies(n, views_number, networkTopology, leaderRotation)

                if args.basic or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "basic",
                        'Faulty': args.faulty,
                        'Latency (ms)': setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights, leaderRotation,
                                           type='basic', numberOfViews=views_number, faulty=args.faulty),
                    })

                if (args.weighted and not args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "weighted",
                        'Faulty': args.faulty,
                        'Latency (ms)': setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights,
                                                                       leaderRotation,
                                                                       type='weighted', numberOfViews=views_number,
                                                                       faulty=args.faulty),
                    })

                if (args.best and not args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "best",
                        'Faulty': args.faulty,
                        'Latency (ms)': setupChainedHotstuffSimulation(n, f, delta, networkTopology, Lphases, weights,
                                                                       leaderRotation,
                                                                       type='best', numberOfViews=views_number,
                                                                       faulty=args.faulty),
                    })

                if (args.weighted and args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "optimal leader",
                        'Faulty': args.faulty,
                        'Latency (ms)': chainedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, views_number, type='weighted',
                                             faulty=args.faulty),
                    })
                if (args.best and args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "optimal leader + best",
                        'Faulty': False,
                        'Latency (ms)': chainedHotstuffBestAndOptimalLeader(n, f, delta, networkTopology, Lphases, views_number),
                    })


# run other experiments on HOTSTUFF
else:
    # gather all data
    csv_filename = os.path.join(data_directory, f'experiment_{timestamp}.csv')

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldNames = ['Simulation Number', 'View Number', 'Type', 'Faulty', 'Latency (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)

        writer.writeheader()

        for simulation_number in range(args.sim):
            for views_number in viewNumbers:
                leaderRotation = getLeaderRotation(n, views_number)
                Lphases = generateExperimentLatencies(n, views_number, networkTopology, leaderRotation)

                if args.basic or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "basic",
                        'Faulty': args.faulty,
                        'Latency (ms)': runWeightedHotstuff(n, f, delta, networkTopology, Lphases, [1] * n, leaderRotation, type="basic",
                                          numberOfViews=views_number, faulty=args.faulty),
                    })

                if (args.weighted and not args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "weighted",
                        'Faulty': args.faulty,
                        'Latency (ms)': runWeightedHotstuff(n, f, delta, networkTopology, Lphases, weights, leaderRotation, type="weighted", faulty=args.faulty,
                                                numberOfViews=views_number),
                    })
                if args.continuous or args.all:
                    (latency, latencyFaulty) = continuousWeightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation, views_number)
                    if args.faulty:
                        latency = latencyFaulty

                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "continuous",
                        'Faulty': args.faulty,
                        'Latency (ms)': latency,
                    })

                if (args.best and not args.lr) or args.all:
                    (latency, latencyFaulty) = weightedHotstuff(n, f, delta, networkTopology, Lphases, leaderRotation,
                                                                views_number)
                    if args.faulty:
                        latency = latencyFaulty

                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "best",
                        'Faulty': args.faulty,
                        'Latency (ms)': latency,
                    })

                if (args.weighted and args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "optimal leader",
                        'Faulty': args.faulty,
                        'Latency (ms)': weightedHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, views_number, faulty=args.faulty),
                    })
                if (args.best and args.lr) or args.all:
                    writer.writerow({
                        'Simulation Number': simulation_number,
                        'View Number': views_number,
                        'Type': "optimal leader + best",
                        'Faulty': args.faulty,
                        'Latency (ms)': weightedBestHotstuffOptimalLeader(n, f, delta, networkTopology, Lphases, weights, views_number, faulty=args.faulty),
                    })
