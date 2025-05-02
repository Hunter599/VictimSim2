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
from map import Map

#------------------- Imports added by students --------------------#
from collections import deque
from vs.abstract_agent import AbstAgent

# Only the four cardinal directions: up, right, down, left
DIRS = [
    (0, -1),  # up
    (1,  0),  # right
    (0,  1),  # down
    (-1, 0),  # left
]

# Action index in the 8-element obstacles tuple returned by check_walls_and_lim()
DIR_TO_INDEX = {
    (0, -1): 0,
    (1,  0): 2,
    (0,  1): 4,
    (-1, 0): 6,
}

FREE = 0  # constant meaning "free to move" from check_walls_and_lim()

#------------------------------------------------------------------#

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


print(">>> LOADING my_test1/explorer.py <<<")

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        # DEBUG: print the time limit you just loaded
        print(f"[{self.NAME}] loaded TLIM = {self.TLIM}")

        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        print(f">>> {self.NAME}::__init__ called, starting at ({self.x},{self.y})")
        
        #------------------- Added by students -------------------#
        # Frontier: cells we’ve discovered but not yet visited
        self.frontier = deque()

        # Visited: cells we have already stepped on
        self.visited = set()

        # Mark the starting cell as already visited
        self.visited.add((self.x, self.y))
        #---------------------------------------------------------#

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())



    def get_next_position(self):
        
        current = (self.x, self.y)

        # Sense environment
        obstacles = self.check_walls_and_lim()
        self.visited.add(current)

        # Expand frontier
        for dx, dy in DIRS:
            idx = DIR_TO_INDEX[(dx, dy)]
            if obstacles[idx] == FREE:
                nbr = (self.x + dx, self.y + dy)
                if nbr not in self.visited and nbr not in self.frontier:
                    self.frontier.append(nbr)

        # If done exploring, go back to base
        if not self.frontier:
            path = self._bfs_path(current, (0, 0))
            return path[1] if len(path) > 1 else current

        # Pick oldest unexplored cell
        target = self.frontier[0]
        path = self._bfs_path(current, target)

        if len(path) > 1:
            return path[1]
        else:
            self.frontier.popleft()
            return self.get_next_position()  # try again

        
        #Tacla's code -- Random roam
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        """
        
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
    
        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[direction] == VS.CLEAR:
                return Explorer.AC_INCR[direction]
        """
    
    def _bfs_path(self, start, goal):
        from collections import deque

        queue = deque([[start]])
        seen = {start}

        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == goal:
                return path

            # expand only the four cardinal neighbors
            for dx, dy in DIRS:
                nbr = (x + dx, y + dy)
                if nbr in seen:
                    continue
                # only walk on cells you’ve already discovered
                if nbr not in self.map.map_data:
                    continue
                # check the stored actions for (x,y)
                _, _, actions = self.map.map_data[(x, y)]
                # map (dx,dy) → its index in the 8-tuple
                idx = DIR_TO_ACTION_INDEX[(dx, dy)]
                # if that direction was free when you sensed it
                if not actions[idx]:
                    continue

                seen.add(nbr)
                queue.append(path + [nbr])

        # if unreachable
        return [start]




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

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
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
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        
        next_pos = self.get_next_position()
        dx = next_pos[0] - self.x
        dy = next_pos[1] - self.y

        result = self.walk(dx, dy)

        if result == 0:  # EXECUTED
            self.x += dx
            self.y += dy

        return True  # keep running
        #Tacla's code
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        
        """
        
        consumed_time = self.TLIM - self.get_rtime()
        if consumed_time < self.get_rtime():
            self.explore()
            return True

        # time to come back to the base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            #input(f"{self.NAME}: type [ENTER] to proceed")
            self.resc.go_save_victims(self.map, self.victims)
            return False

        self.come_back()
        return True
        """
