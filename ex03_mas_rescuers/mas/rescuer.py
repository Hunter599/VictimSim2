##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by K-Means
### - definition of a sequence of rescue of victims of a cluster using a multi-objective Genetic Algorithm
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
from .map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from .bfs import BFS
from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans

# Imports for the Genetic Algorithm
from deap import base, creator, tools, algorithms


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        if not os.path.exists("./clusters"):
            os.makedirs("./clusters")
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'x', 'y', 'grav', 'classe'])
            for vic_id, values in cluster.items():
                x, y = values[0]
                vs = values[1]
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        if not os.path.exists("./clusters"):
            os.makedirs("./clusters")
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'x', 'y', 'grav', 'classe'])
            for vic_id, values in sequence.items():
                x, y = values[0]
                vs = values[1]
                writer.writerow([vic_id, x, y, vs[6], vs[7]])


    def cluster_victims(self):
        NB_OF_CLUSTERS = 4
        if not self.victims:
            return [{} for _ in range(NB_OF_CLUSTERS)]
        victim_ids = list(self.victims.keys())
        victim_coords = np.array([self.victims[vic_id][0] for vic_id in victim_ids])
        num_clusters_to_form = min(len(victim_ids), NB_OF_CLUSTERS)
        if num_clusters_to_form == 0:
            return [{} for _ in range(NB_OF_CLUSTERS)]
        kmeans = KMeans(n_clusters=num_clusters_to_form, random_state=0, n_init=10)
        kmeans.fit(victim_coords)
        clusters = [{} for _ in range(NB_OF_CLUSTERS)]
        for i, label in enumerate(kmeans.labels_):
            vic_id = victim_ids[i]
            clusters[label][vic_id] = self.victims[vic_id]
        print(f"{self.NAME}: Clustering complete. {len(self.victims)} victims partitioned into {num_clusters_to_form} clusters.")
        for i, cluster in enumerate(clusters):
            print(f"  - Cluster {i+1} has {len(cluster)} victims.")
        return clusters

    def predict_severity_and_class(self):
        """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
            This method should add the vital signals(vs) of the self.victims dictionary with these two values.

            This implementation assigns random values to both, severity value and class"""
        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 99.9)          # to be replaced by a regressor 
            severity_class = random.randint(1, 4)               # to be replaced by a classifier
            values[1].extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )

    def sequencing(self):
        """
        Optimized version of the Multi-Objective Genetic Algorithm.
        It pre-calculates all path distances to avoid redundant BFS calls,
        significantly speeding up the fitness evaluation.
        """
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        new_sequences = []

        for cluster in self.clusters:
            if not cluster:
                new_sequences.append({})
                continue

            victim_ids = list(cluster.keys())
            num_victims = len(victim_ids)
            if num_victims == 0:
                new_sequences.append({})
                continue
            
            # --- OPTIMIZATION: Pre-calculate and cache all distances ---
            dist_cache = {}
            points = {vic_id: cluster[vic_id][0] for vic_id in victim_ids}
            points['base'] = (0, 0)
            point_ids = list(points.keys())

            for i in range(len(point_ids)):
                for j in range(i, len(point_ids)):
                    p1_id = point_ids[i]
                    p2_id = point_ids[j]
                    _, time = bfs.search(points[p1_id], points[p2_id], self.plan_rtime)
                    dist_cache[(p1_id, p2_id)] = time
                    dist_cache[(p2_id, p1_id)] = time
            
            # --- GA registration for this specific cluster ---
            toolbox.register("indices", random.sample, range(num_victims), num_victims)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate_tour(individual_indices):
                total_time = 0.0
                total_severity = 0.0
                
                first_vic_id = victim_ids[individual_indices[0]]
                total_severity += cluster[first_vic_id][1][6]
                
                # 1. From base to first victim
                time = dist_cache.get(('base', first_vic_id), -1)
                if time < 0: return sys.maxsize, 0
                total_time += time

                # 2. Between victims
                for i in range(num_victims - 1):
                    start_vic_id = victim_ids[individual_indices[i]]
                    end_vic_id = victim_ids[individual_indices[i+1]]
                    total_severity += cluster[end_vic_id][1][6]
                    time = dist_cache.get((start_vic_id, end_vic_id), -1)
                    if time < 0: return sys.maxsize, 0
                    total_time += time

                # 3. From last victim back to base
                last_vic_id = victim_ids[individual_indices[-1]]
                time = dist_cache.get((last_vic_id, 'base'), -1)
                if time < 0: return sys.maxsize, 0
                total_time += time
                
                if total_time > self.plan_rtime:
                    return sys.maxsize, 0
                return total_time, total_severity

            toolbox.register("evaluate", evaluate_tour)
            toolbox.register("mate", tools.cxOrdered)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
            toolbox.register("select", tools.selNSGA2)

            # --- RUN THE GA with optimized parameters ---
            population = toolbox.population(n=80) # Slightly smaller population
            NGEN = 40 # Fewer generations needed due to faster evaluation
            CXPB, MUTPB = 0.8, 0.2

            algorithms.eaMuPlusLambda(population, toolbox, mu=80, lambda_=80,
                                      cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                      stats=None, halloffame=None, verbose=False)
            
            best_individual = tools.selBest(population, 1)[0]
            
            ordered_sequence = {victim_ids[index]: cluster[victim_ids[index]] for index in best_individual}
            new_sequences.append(ordered_sequence)

        self.sequences = new_sequences
        print(f"{self.NAME}: Sequencing complete using Optimized Multi-Objective GA.")
        
    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        if not self.sequences:
            return
        
        # we consider only the first sequence (the simpler case)
        sequence = self.sequences[0]
        if not sequence: return

        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            if plan:
                self.plan = self.plan + plan
                self.plan_rtime = self.plan_rtime - time
                start = goal
            else:
                print(f"{self.NAME}: Could not plan path to {vic_id}, stopping plan.")
                break
        
        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        if plan:
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """
        self.received_maps += 1
        self.map.update(explorer_map)
        self.victims.update(victims)
        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            clusters_of_vic = self.cluster_victims()
            
            for i, cluster in enumerate(clusters_of_vic):
                if cluster:
                    self.save_cluster_csv(cluster, i+1)
            
            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent
            
            # Assign the cluster the master agent is in charge of
            self.clusters = [clusters_of_vic[0]]
            
            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map
            
            # Calculate the sequence of rescue for each agent
            self.sequences = self.clusters
            
            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                if not rescuer: continue
                rescuer.sequencing()
                for j, sequence in enumerate(rescuer.sequences):
                    if not sequence: continue
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)
         
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """
        # No more actions to do
        if self.plan == []:
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
        return True
