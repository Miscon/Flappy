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
        """ 
            player_y
            next_pipe_top_y
            next_pipe_dist_to_player
            player_vel
        """

        return (state['player_y'], state['player_vel'], state['next_pipe_dist_to_player'], state['next_pipe_top_y'])

    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """

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


    def draw_plots(self):

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        while True:
            try:
                ax.cla()
                self.plot_learning_curve(ax)                

                plt.pause(5)
            except:
                pass


    def plot_learning_curve(self, ax):
        
        df = pd.read_csv("{}/scores.csv".format(self.name))

        ax.plot(df["frame_count"], df["avg_score"])
        ax.fill_between(df["frame_count"], df["interval_upper"], df["interval_lower"], alpha=0.5)

        ax.set_xlabel("Number of frames")
        ax.set_ylabel("Average score")
        ax.set_title("{}\nAverage score from 10 games".format(self.name.replace("_", " ")))


name = "linear_approximation"
agent = LinearApproximation(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
