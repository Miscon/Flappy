from flappy_agent import FlappyAgent

from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import sys
import matplotlib.pyplot as plt

class IfElse(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.jumped_last = False
        self.inside_jump = False
  

    def get_argmax_a(self, state):
        # Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).
        pipe_center_y = state["next_pipe_top_y"] + 45
        next_pipe_center_y = state["next_next_pipe_top_y"] + 45
        player_y = state["player_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]
        distance_to_next_pipe = state["next_pipe_dist_to_player"]
        player_vel = state["player_vel"]
        
        action = 0
        
        
        # when distance is less than 20 the bird has
        # actually moved past the current pipe
        # so make the next pipe is the current pipe
        if distance_to_pipe < 10:
            distance_to_pipe = distance_to_next_pipe
            pipe_center_y = next_pipe_center_y

        # get y difference between bird and pipe opening
        difference = player_y - pipe_center_y 

        # align to the current pipe
        if difference < 0:
            action = 1
 
        if distance_to_pipe < 100 and distance_to_pipe >= 0 or difference < 30: # if inside pipe
           # if self.inside_jump:
           #     action = 0
           #     self.inside_jump = False
            if self.jumped_last:
                action = 0
                self.jumped_last = False
                self.inside_jump = True 
                return action
            
            self.inside_jump = action == 0
            self.jumped_last = action == 0
            #import time
            #time.sleep(2)

            return action
        else: # if outside pipe
            if self.jumped_last or self.inside_jump:
                action = 1

        self.inside_jump = False
        self.jumped_last = action == 0
        return action


    def run(self, arg):
        if arg == "train":
            print("Invalid argument, if_else agent can not train, use 'play' or 'score'")
        elif arg == "play":
            self.play()
        elif arg == "score":
            self.score(False, 10)
        else:
            print("Invalid argument, use 'play'")


name = "if_else"
agent = IfElse(name)

agent.run(sys.argv[1]) # Use 'train' or 'play'
