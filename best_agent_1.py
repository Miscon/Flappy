from flappy_agent import FlappyAgent

from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import sys
import math
import matplotlib.pyplot as plt




class BestAgent1(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.player_pipe_difference_bins = np.linspace(0, 100, 11)
        self.pipe_next_pipe_difference_bins = np.linspace(-158, 158, 5)
        self.distance_to_pipe_bins = np.linspace(0, 65, 4)

        
    def map_state(self, state):

        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]
        next_pipe_top_y = state["next_next_pipe_top_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]

        player_pipe_difference = player_y - pipe_top_y
        pipe_next_pipe_difference = pipe_top_y - next_pipe_top_y

        player_pipe_difference_bin = np.digitize([player_pipe_difference], self.player_pipe_difference_bins)[0]
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], self.pipe_next_pipe_difference_bins)[0]
        distance_to_pipe_bin = np.digitize([distance_to_pipe], self.distance_to_pipe_bins)[0]

        if player_vel <= -8:
            player_vel = -8

        return (player_pipe_difference_bin, player_vel, pipe_next_pipe_difference_bin, distance_to_pipe_bin)

    
    def observe(self, s1, a, r, s2, end):

        s1 = self.map_state(s1)
        s2 = self.map_state(s2)
        
        # count for graphs
        self.update_counts((s1, a))

        self.learn_from_observation(s1, a, r, s2)

        # Count episodes
        if end:
            self.episode_count += 1


    def learn_from_observation(self, s1, a, r, s2):        
        
        # Get state values
        Qs1a = self.Q.get((s1, a))
        if Qs1a is None:
            Qs1a = self.get_initial_return_value(s1, a)

        # Calculate return
        G = r + self.gamma * self.get_max_a(s2) 

        # Update Q table
        self.Q[(s1, a)] = Qs1a + self.lr * (G - Qs1a) # update rule


    def state_to_dict(self, state, action, G):
        return {"y_difference":state[0],
                "next_pipe_dist_to_player":state[3],
                "action":self.get_argmax_a(state),
                "return":G,
                "count_seen":self.s_a_counts[(state, action)]}


name = "best_agent_1"

agent = BestAgent1(name)

try:
    with open("{}/agent.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    if sys.argv[1] == "plot":
        print("No data available to plot")
        quit()
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train', 'play', 'score' or 'plot'
