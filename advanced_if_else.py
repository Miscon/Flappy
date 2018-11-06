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

class AdvancedIfElse(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.count = 0
        self.split = 5
  

    def get_argmax_a(self, state):


        # Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).
        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_center_y = state["next_pipe_top_y"] + 55
        next_pipe_center_y = state["next_next_pipe_top_y"] + 55
        distance_to_pipe = state["next_pipe_dist_to_player"]
        next_distance_to_pipe = state["next_next_pipe_dist_to_player"]


        # next pipe is the current pipe
        if distance_to_pipe < 15:
            distance_to_pipe = next_distance_to_pipe
            pipe_center_y = next_pipe_center_y

        action = 0

        # if player is below the center of the pipe
        difference = player_y - pipe_center_y
        if difference < 0:
            action = 1

        # if inside the pipe, do double jumps
        if distance_to_pipe < 100:
            # if player jump last action
            if player_vel < 0:
                action = 0
        # else do jump, noop, jump to jump higher
        else:
            # if last action was jump, don't jump right away
            if player_vel < 0:
                action = 1


        return action


    def run(self, arg):
        if arg == "train":
            print("Invalid argument, if_else agent can not train, use 'play' or 'score'")
        elif arg == "play":
            self.play()
        elif arg == "score":
            self.score(False, 100)
        else:
            print("Invalid argument, use 'play'")


name = "advanced_if_else"
agent = AdvancedIfElse(name)

agent.run(sys.argv[1]) # Use 'train' or 'play'
