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


# Action increments: index → (dx, dy)
AC_INCR = AbstAgent.AC_INCR


# We only want the four orthogonal moves:
DIRS = [
    (0, -1),  # up
    (1,  0),  # right
    (0,  1),  # down
    (-1, 0),  # left
]

# Map each of those (dx,dy) into the correct slot in the 8-tuple that
# check_walls_and_lim() returns (it matches AbstAgent.AC_INCR indices)
DIR_TO_ACTION_INDEX = {
   0: (0, -1),  # up
   2: (1,  0),  # right
   4: (0,  1),  # down
   6: (-1, 0),  # left
}
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



class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
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
        
        # Create reverse mapping from direction to action index
        ACTION_INDEX_FROM_DIR = {
            (0, -1): 0,   # up
            (1, 0): 2,    # right
            (0, 1): 4,    # down
            (-1, 0): 6    # left
        }
        
        obstacles = self.check_walls_and_lim()
        current_pos = (self.x, self.y)
        
        # 1. First priority: unvisited adjacent cells
        unvisited = []
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:  # Right > Down > Left > Up
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            
            # Skip out-of-bounds immediately
            if not (0 <= new_x < 90 and 0 <= new_y < 90):
                continue
                
            # Get the correct action index for this direction
            dir_index = ACTION_INDEX_FROM_DIR.get((dx, dy), None)
            if dir_index is None or obstacles[dir_index] != VS.CLEAR:
                continue
                
            if (new_x, new_y) not in self.map.map_data:
                unvisited.append((dx, dy))
        
        if unvisited:
            return unvisited[0]

        # 2. Second priority: random valid move
        valid_moves = []
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            dir_index = ACTION_INDEX_FROM_DIR.get((dx, dy), None)
            
            if (dir_index is not None and 
                obstacles[dir_index] == VS.CLEAR and
                0 <= new_x < 90 and 
                0 <= new_y < 90 and
                self.map.map_data.get((new_x, new_y), (VS.CLEAR,))[0] != VS.OBST_WALL):
                valid_moves.append((dx, dy))
        
        return random.choice(valid_moves) if valid_moves else (0, 0)


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
        
        # Single movement attempt per call
        dx, dy = self.get_next_position()
        
        # Original movement code below
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            bumped_x = self.x + dx
            bumped_y = self.y + dy
            if 0 <= bumped_x < 90 and 0 <= bumped_y < 90:
                self.map.add((bumped_x, bumped_y), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        if result == VS.EXECUTED:
            self.walk_stack.push((dx, dy))
            self.x += dx
            self.y += dy
        
            # Victim handling
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

            # Difficulty calculation
            difficulty = (rtime_bef - rtime_aft)
            difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG

            # Map update
            self.map.add((self.x, self.y), 
                        difficulty, 
                        seq, 
                        self.check_walls_and_lim())

        return
        
        """
        # Keep original code below completely unchanged
        dx, dy = self.get_next_position()
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)

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
        """
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
        """BFS-based decision-making at every tick."""

        next_pos = self.get_next_position()
        dx = next_pos[0] - self.x
        dy = next_pos[1] - self.y

        result = self.walk(dx, dy)

        print(f"[{self.NAME}] @ ({self.x},{self.y}) → ({dx},{dy}) → walk result: {result}")

        if result == 0:  # move succeeded
            self.x += dx
            self.y += dy

        return True


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
