import hashlib
import random

## define some data
DATA = "SOME COMMAND"

class Network:
    def __init__(self, replicas):
        self.nodes = replicas

    def broadcast(self, leader, message):
        if message.type == "PREPARE":
            for node in self.nodes:
                if node.id != leader.id:
                    ## skip the verification part and accept anyways

                    # EACH REPLICA SENDS PREPARE MESSAGE
                    leader.votes.append(Message("PREPARE", message.justify.node, message.viewNumber, None))

                    # EACH REPLICA ALSO SENDS PRE-COMMIT MESSAGE -> 2nd check that each replica receives PREPARE message
                    # leader.votes.add(Message("PRE-COMMIT", message.node, message.viewNumber, None))

        if message.type == "PRE-COMMIT":
            for node in self.nodes:
                if node.id != leader.id:
                    ## skip the verification part and accept anyways

                    # EACH REPLICA SENDS PREPARE MESSAGE
                    leader.votes.append(Message("COMMIT", message.justify.node, message.viewNumber, None))

        if message.type == "COMMIT":
            for node in self.nodes:
                if node.id != leader.id:
                    # each node should execute the command here
                    continue;



class Block:
    def __init__(self, height, parent_hash, data):
        self.height = height
        self.parent_hash = parent_hash
        self.data = data
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha = hashlib.sha256()
        sha.update(str(self.height).encode('utf-8') +
                   str(self.parent_hash).encode('utf-8') +
                   str(self.data).encode('utf-8'))
        return sha.hexdigest()


class QuorumCertificate:
    def __init__(self, type, viewNumber, node):
        self.type = type
        self.viewNumber = viewNumber
        self.node = node
        self.signature = None

    def compute_signature(self, votes):
        signatures = [Message.message_signature(vote) for vote in votes]
        self.signature = ''.join(signatures)


class Message:
    ## send also the viewnumber such that we mimic the automatic stamping of the messages with the currentView
    def __init__(self, type, node, viewNumber, qc):
        self.type = type
        self.node = node
        self.viewNumber = viewNumber
        self.justify = qc

    ## some signature for the message to be used for quorum certificates
    #### DRAFT
    def message_signature(self):
        sha = hashlib.sha256()
        sha.update(str(self.type).encode('utf-8') +
                   str(self.node).encode('utf-8') +
                   str(self.viewNumber).encode('utf-8'))
        return sha.hexdigest()


class Replica:
    def __init__(self, id):
        self.id = id
        self.blocks = []
        self.pending_blocks = []
        self.votes = []
        self.committed_block = None

    def generate_block(self, data):
        if len(self.blocks) == 0:
            parent_hash = "genesis"
            height = 1

        else:
            last_block = self.blocks[-1]
            parent_hash = last_block.hash
            height = last_block.height + 1

        new_block = Block(height, parent_hash, data)
        self.pending_blocks.append(new_block)

    def propose_block(self):
        if len(self.pending_blocks) > 0:
            proposed_block = self.pending_blocks.pop(0)
            self.blocks.append(proposed_block)
            return proposed_block

        return None

    def receive_proposed_block(self, proposed_block):
        if len(proposed_block.data) > 0:
            self.pending_blocks.append(proposed_block)


class HotStuff:
    def __init__(self, network, nodes, faulty_replicas):
        self.network = network
        self.nodes = nodes
        self.faulty_replicas = faulty_replicas
        self.leader = None

    def set_leader(self, node):
        self.leader = node

    def simulate_round(self, maxViews):
        pending_newview = []
        for current_view in range(maxViews):
            # Set leader
            leader = random.choice(nodes)
            hotstuff.set_leader(leader)

            ### PREPARE PHASE
            if len(pending_newview) == 0:
                # genesis block needs to be proposed and formed
                leader.generate_block(DATA)
                currentProposal = leader.propose_block()
                highQC = QuorumCertificate("NEW-VIEW", current_view, leader)

                ## broadcast prepare and then receive PREPARE votes from the replicas
                self.network.broadcast(leader, Message("PREPARE", currentProposal, current_view, highQC))
            else:
                maximumViewNumber = 0
                for message in pending_newview:
                    if message.justify.viewNumber > maximumViewNumber:
                        maximumViewNumber = message.justify.viewNumber
                        highQC = message.justify

                currentProposal = highQC.node.propose_block()

                ## broadcast prepare and then receive PREPARE votes from the replicas
                self.network.broadcast(leader, Message("PREPARE", currentProposal, current_view, highQC))

            ## PRE-COMMIT PHASE

            ## leader counts majority
            majority = self.nodes - self.faulty_replicas
            counter = 0

            # print(len(leader.votes))
            # print(majority)

            votes = []
            while counter < majority:
                votes.append(leader.votes[counter])
                counter += 1

            prepareQC = QuorumCertificate("PREPARE", current_view, leader)
            prepareQC.compute_signature(votes)

            ## reset leader votes
            leader.votes = []

            ## each replica received PREPARE message and now updates prepareQC with its information
            ## skip this step since it's hard to replicate without network connection + we send the same highQC to all replicas hence prepareQC will remain the same

            ## MANUALLY SEND PRE-COMMIT FROM EACH REPLICA
            for node in self.network.nodes:
                if node.id != leader.id:
                    ## skip the verification part and accept anyways
                    # EACH REPLICA ALSO SENDS PRE-COMMIT MESSAGE -> 2nd check that each replica receives PREPARE message
                    leader.votes.append(Message("PRE-COMMIT", highQC.node, current_view, None))

            ## COMMIT PHASE
            majority = self.nodes - self.faulty_replicas
            counter = 0

            votes = []
            while counter < majority:
                votes.append(leader.votes[counter])
                counter += 1

            precommitQC = QuorumCertificate("PRE-COMMIT", current_view, leader)
            precommitQC.compute_signature(votes)

            ## reset leader votes
            leader.votes = []

            ## BROADCAST AS PART OF PRE-COMMIT PHASE
            ## broadcast pre-commit and then receive COMMIT votes from the replicas
            self.network.broadcast(leader, Message("PRE-COMMIT", None, current_view, prepareQC))

            ## in this case they are the same
            lockedQC = precommitQC


            ## DECIDE PHASE
            majority = self.nodes - self.faulty_replicas
            counter = 0

            votes = []
            while counter < majority:
                votes.append(leader.votes[counter])
                counter += 1

            commitQC = QuorumCertificate("COMMIT", current_view, leader)
            commitQC.compute_signature(votes)

            ## reset leader votes
            leader.votes = []

            ## BROADCAST AS PART OF COMMIT PHASE
            self.network.broadcast(leader, Message("COMMIT", None, current_view, precommitQC))

            ## FINAL BROADCAST DECIDE AND TRIGGER NEWVIEW
            self.network.broadcast(leader, Message("DECIDE", None, current_view, commitQC)) ## nothing happens here

            for _ in self.network.nodes:
                pending_newview.append(Message("NEW-VIEW", None, current_view, lockedQC))


## PARAMETERS THAT CAN BE TWEAKED
f = 1  # max num of faulty replicas
delta = 0  # additional replicas

n = 3 * f + 1 + delta  # total num of replicas
rounds = 10
maxViews = 5

# Simulate rounds
for _ in range(rounds):
    # Create nodes
    nodes = [Replica(i) for i in range(1, 5)]

    # Create network
    network = Network(nodes)
    hotstuff = HotStuff(network, n, f)

    hotstuff.simulate_round(maxViews)
