import argparse
import os, datetime

import leaderRotation_impactAnalysis
import paper_experiment_chained
import paper_experiment_hotstuff

def create_directory(base_directory="results"):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    return base_directory

parser = argparse.ArgumentParser(description='Experiment simulations for Hotstuff')
parser.add_argument("--chained", action="store_true", default=False, help="Run experiments with Chained Hotstuff.")
parser.add_argument("--paper", action="store_true", default=False, help="Run experiments from paper.")
parser.add_argument("--lr", action="store_true", default=False, help="Run analysis on leader rotation impact.")
args = parser.parse_args()


output_directory = create_directory()
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

if args.paper:
    if args.chained:
        paper_experiment_chained.experiment(output_directory, timestamp)
    else:
        paper_experiment_hotstuff.experiment(output_directory, timestamp)

if args.lr:
    leaderRotation_impactAnalysis.experiment(output_directory, timestamp)
