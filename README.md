# Using Weighted Voting to Optimise Streamlined Blockchain Consensus Algorithms



## Overview

This is the accompanying repository to the paper **"Using Weighted Voting to Optimise Streamlined Blockchain Consensus Algorithms"**, developed for my Bachelor's Thesis at TU Delft. This research project was conducted as part of the CSE3000 course of the Computer Science and Engineering degree program, under the close supervision of [Jérémie Decouchant](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/software-technology/distributed-systems/people/jeremie-decouchant) and [Rowdy Chotkan](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/software-technology/distributed-systems/people/rowdy-chotkan).

The corresponding research paper is available [here]().


## Current status
Software development ended as of _26th of June, 2024 (Thesis defense day)._

## Table of contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Contact](#contact)

## Introduction

This project investigates the impact of weighted voting on streamlined consensus algorithms. Inspired by [AWARE's](https://doi.org/10.48550/arXiv.2011.01671) established self-monitoring (deterministic latency prediction) and self-optimising (leader relocation and weight distribution tuning) mechanism, this research applies weighted voting on the representative [Hotstuff](https://github.com/asonnino/hotstuff}).

The focus of this study lies on:

1. Developing two latency prediction models for **Weighted Hotstuff** and **Weighted Chained Hotstuff**, respectively.
2. Analysing the impact of vote power assignment **(Best Assigned Weighted)** and leader rotation **(Optimal Leader Rotation Weighted)** optimisations by employing _Simulated Annealing_ algorithms.
3. Introducing a generalisation from the discrete weighting paradigm (a novel continuous weighting scheme) through a Simulated Annealing approach in **Continuous Weighted Hotstuff.**


In short, the results provided in this research, together with the novel ideas described, are a founding base for the study of weighted voting in streamlined algorithms and its shift from the discrete model.

## Installation
To set up the project locally, follow these steps:
1. Clone the repository
2. Run the command below in terminal, in this way you create a virtualenv named `venv`, activate it and install all the required dependencies for running the project (note that we use python version 3.10).

    `virtualenv venv && source venv/bin/activate && pip3 install -r requirements.txt`

**The project should function properly now.**

## Usage
For running experiments you will use the `experiments.py` script, using commands of the following form, with various options that are explained below.

`python3 experiments.py  --paper --continuous`

### Experiment options
You can use the following options to change some of the parameters for running custom experiments.

#### Results from paper
By running this commands the corresponding results can be found in root project under `./results/figures` and `./results/data`.
- `--paper` runs `paper_experiment_hotstuff`
- `--paper --chained` runs `paper_experiment_chained`
- `--paper --continuous` runs `paper_experiment_continuous`
- `--paper --lr` runs `paper_leader_rotation`

### Custom experiments
By running self-tailored experiments the results will be gathered in a **CSV file** in `./results/data` for further analysis and processing.
- `--chained` run experiments with Chained Hotstuff **(Without this flag, experiments run with Hotstuff by default!)**
- `--sim` to change the number of protocol simulations (by default 1)
- `--views-lower-bound` and `--views-upper-bound` to set for how many views we want to run a protocol for, they constitute an interval as we can run experiments over multiple views too (by default both 1)
- `--f` to change the number of arbitrary failures the system can withstand
- `--delta` to change the number of additional replicas in the system
- `--faulty` to run experiments in faulty scenarios (the f replicas holding highest weight are considered idle)
- `--all` runs all protocols, but you can also specify specific ones:
  - `--basic` runs unweighted protocol
  - `--weighted` runs weighted protocol
  - `--best` runs Best Assigned Weighted protocol variant
  - `--lr`runs Optimal Leader Rotation Weighted protocol variant
  - `--best --lr` runs (Optimal Leader Rotation + Best Assigned) protocol variant
  - `--continuous` runs Continuous Weighted Hotstuff

Note that the network setup on which we run experiments can be also tweaked by using the `--network-setup` option. The framework offers the following variants (see more on `experimental_utils`:
- zero (default value) - run experiments with matrix of withing clusters latency generated randomly between `0ms and 400ms`
- one - run experiments on the network environment used in the paper (see `./results/figures/clusters`)
- two - run experiments with `f = 2` on custom network topology with data retrieved from [cloudping](https://www.cloudping.co/grid/latency/timeframe/1D)
- three - run experiments with `f = 3` on custom network topology with data retrieved from [cloudping](https://www.cloudping.co/grid/latency/timeframe/1D)

## Project structure

The main file through which you can run multiple experiments is `experiments.py`. However, this repository consists of multiple files which support the research of weighted voting on streamlined algorithms.

### Latency prediction + Simulated Annealing  models
1. `weighted_hotstuff`
2. `chained_weighted_hotstuff`

### Experiments included in paper
1. `paper_leader_rotation` - analysis of **leader rotation** impact on the latency of **Hotstuff protocol** run
2. `paper_experiment_hotstuff` - non-faulty and faulty simulation of **Weighted Hotstuff** and variants for computing **average latency per view** over multiple views
3. `paper_experiment_chained` - non-faulty and faulty simulation of **Weighted Chained Hotstuff** and variants for computing **average latency per view** over multiple views
4. `paper_experiment_continuous` - Best Assigned vs Continuous Weighted Hotstuff, latency difference analysis over 1000 simulations

### Additional experiments 
These files should be run directly from the IDE, not by using the experimental framework created.
1. `experiments_hotstuff` - 4 experiments on assessing Weighted Hotstuff behaviour + analysis of Continuous Weighted Hotstuff convergence time for multiple f values
2. `experimemts_chained_hotstuff` - 3 experiments of Weighted Chained Hotstuff and its optimisation variants

### Other files
1. `hotstuff` - skeleton of Basic Hotstuff implementation in python to observe protocol's communication phases 
2. `experimental_utils` - utils used by the developed latency prediction models
3. `map` - file generating the geographical map of clusters used in the paper experiments


## Contact
For any direct questions please feel free to contact [Diana Micloiu](mailto:d.micloiu@yahoo.com). For further questions or collaboration inquires with the Professor and Supervisor you can contact the [Data Intensive group](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/software-technology/distributed-systems/contact).
