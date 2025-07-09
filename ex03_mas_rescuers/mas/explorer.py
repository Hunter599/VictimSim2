# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from .map import Map
# NEW: Import BFS to allow for intelligent pathfinding back to base
from .bfs import BFS
from collections import deque

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc,exp_id):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.direction=[]
        self.id=exp_id           # explorer's id
        self.bump=0
        self.auxMap = {}     # a map to store the visited cells
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        
        # NEW: Add a BFS instance for pathfinding
        self.bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        # NEW: Add attributes for the robust return trip
        self.returning_to_base = False
        self.path_to_base = deque()


        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        if exp_id == 1:
            self.direction = [4,5,3,6,2,0,7,1]
        elif exp_id == 2:
            self.direction = [2,3,1,4,0,6,5,7]
        elif exp_id == 3:
            self.direction = [0,1,7,2,6,4,3,5]
        elif exp_id == 4:
            self.direction = [6,7,5,0,4,2,3,1]

    # This function was not modified
    def verifica_envolta(self):
        for i in range(8):
            if self.auxMap.get((self.x+Explorer.AC_INCR[i][0], self.y+Explorer.AC_INCR[i][1])) is None:
                return True
        return False

    # This function was not modified
    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
    
        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            # direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            #------------------- Added by students -------------------#
            direction = self.direction[self.bump]

            #---------------------------------------------------------#
            comando= Explorer.AC_INCR[direction]
            item=self.map.get((self.x+comando[0], self.y+comando[1]))
            if self.verifica_envolta():
                if(obstacles[direction] == VS.WALL or obstacles[direction] == VS.END):
                    self.auxMap[self.x+comando[0], self.y+comando[1]]=-1
                item=self.auxMap.get((self.x+comando[0], self.y+comando[1]))
                if self.bump >=7:
                    dx, dy = self.walk_stack.pop()
                    dx = dx * -1
                    dy = dy * -1
                    return(dx, dy)
                elif obstacles[direction] == VS.CLEAR and item is None :
                    self.auxMap[self.x+comando[0], self.y+comando[1]]=1
                    self.walk_stack.push((comando[0], comando[1]))
                    return Explorer.AC_INCR[direction]
                else:
                    print(item)
                    self.bump+=1
            else:
                print("oi")
                dx, dy = self.walk_stack.pop()
                dx = dx * -1
                dy = dy * -1
                return(dx, dy)

    # This function was not modified
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")
            self.bump += 1

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.bump = 0
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        """
        This function is now obsolete and is not used.
        The return trip is handled by the new logic in deliberate().
        """
        pass
        
    def deliberate(self) -> bool:
        """ 
        The agent chooses the next action. The simulator calls this
        method at each cycle. This version implements a robust return-to-base
        mechanism using BFS pathfinding.
        """
        # NEW: Check if the agent is already in the process of returning to base
        if self.returning_to_base:
            # NEW: If there are still steps in the path home, execute the next one
            if self.path_to_base:
                dx, dy = self.path_to_base.popleft()
                self.walk(dx, dy)
                self.x += dx
                self.y += dy
                return True
            else:
                # NEW: The path is empty, meaning the agent has arrived at the base
                print(f"{self.NAME}: Arrived at base. Syncing with rescuer.")
                self.resc.sync_explorers(self.map, self.victims)
                return False # End execution

        # The original logic to decide when to start returning
        time_tolerance = 2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ
        if self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # NEW: Time to go home. Instead of backtracking, calculate a safe path.
        print(f"{self.NAME}: Time to return to base. Calculating path...")
        self.returning_to_base = True # NEW: Set the flag to indicate return mode
        
        # NEW: Use BFS to find the best path from the current position to the base
        path, _ = self.bfs.search((self.x, self.y), (0, 0), self.get_rtime())
        
        if path:
            # NEW: If a path is found, store it
            self.path_to_base = deque(path)
            print(f"{self.NAME}: Path to base calculated. Starting return trip.")
        else:
            # NEW: If no path is found, the agent is trapped
            print(f"{self.NAME}: Could not find a path to base. Agent is trapped.")
            return False # End execution
        
        return True
