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
import math

class BasicIfElse(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.count = 0
        self.split = 5


    def get_argmax_a(self, state):
        pipe_center_y = state["next_pipe_top_y"] + 70
        next_pipe_center_y = state["next_next_pipe_top_y"] + 70
        player_y = state["player_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]

        player_y = state["player_y"]



        action = 0
        
        if distance_to_pipe < 10:
            distance_to_pipe = state["next_next_pipe_dist_to_player"]
            pipe_center_y = state["next_next_pipe_top_y"] + 55

        difference = player_y - pipe_center_y

        if difference < 0:
            action = 1

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


name = "basic_if_else"
agent = BasicIfElse(name)

agent.run(sys.argv[1]) # Use 'train' or 'play'
