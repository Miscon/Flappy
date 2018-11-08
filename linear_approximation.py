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

class LinearApproximation(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.lr = 0.00001
        self.episode = []
        self.weights = [np.array([0]*4, dtype=np.float64), np.array([0]*4, dtype=np.float64)] # [0 flap, 0 noop]
  
    
    def map_state(self, state):

        return (state['player_y'], state['player_vel'], state['next_pipe_dist_to_player'], state['next_pipe_top_y'])

    
    def observe(self, s1, a, r, s2, end):

        s1 = self.map_state(s1)
        self.episode.append((s1, a, r))
        
        # count for graphs
        self.update_counts((s1, a))

        # 2. if reached the end of episode, learn from it, then reset
        if end:
            self.learn_from_episode(self.episode)
            self.episode_count += 1
            self.episode = []


    def get_argmax_a(self, state):

        features = np.array(state, dtype=np.float64)

        G0 = np.dot(self.weights[0], features)
        G1 = np.dot(self.weights[1], features)

        if G0 > G1:
            return 0
        else:
            return 1   


    def learn_from_episode(self, episode):        
        
        G = 0
        for s, a, r in reversed(episode):
            G = r + self.gamma * G
            features = np.array(s, dtype=np.float64)
            self.weights[a] = self.weights[a] - self.lr * (np.dot(self.weights[a], features) - G) * features


    def draw_plots(self, once=False):

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        while True:
            try:
                ax.cla()
                self.plot_learning_curve(ax)                

                if once:
                    plt.show()
                    break
                else:
                    plt.pause(5)
            except:
                pass


name = "linear_approximation"
agent = LinearApproximation(name)

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
