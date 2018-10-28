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
  

    def get_argmax_a(self, state):

        pipe_center_y = state["next_pipe_top_y"] + 65
        next_pipe_center_y = state["next_next_pipe_top_y"] + 65
        player_y = state["player_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]
        
        action = 0
        
        difference = player_y - pipe_center_y
        if distance_to_pipe < 10:
            difference = player_y - next_pipe_center_y
        if difference < 0:
            action = 1

        return action


    def run(self, arg):
        if arg == "train":
            print("Invalid argument, if_else agent can not train, use 'play'")
        elif arg == "play":
            self.play()
        else:
            print("Invalid argument, use 'play'")


name = "if_else"
agent = IfElse(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
