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

class StateAggregatedMonteCarlo(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.episode = []
        self.gamma = 0.95
  
    
    def map_state(self, state):

        player_pipe_difference = state['player_y'] - state['next_pipe_top_y']
        pipe_next_pipe_difference = state['next_pipe_top_y'] - state['next_next_pipe_top_y']
        velocity = state['player_vel']
        dist_to_pipe = state['next_pipe_dist_to_player']

        step = int((203+230)/15) + 1
        bins = np.arange(-202, 230, step)
        player_pipe_difference_bin = np.digitize([player_pipe_difference], bins)[0]

        step = int((157+157)/15) + 1
        bins = np.arange(-157, 157, step)
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], bins)[0]

        step = int((16+10)/15) + 1
        bins = np.arange(-16, 10, step)
        velocity_bin = np.digitize([velocity], bins)[0]

        step = int((309)/15) + 1
        bins = np.arange(0, 309, step)
        dist_to_pipe_bin = np.digitize([dist_to_pipe], bins)[0]

        return (player_pipe_difference_bin, velocity_bin, dist_to_pipe_bin, pipe_next_pipe_difference_bin)

    
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


    def learn_from_episode(self, episode):        
        
        # instead of append G to returns, use the update rule
        G = 0
        for s, a, r in reversed(episode):
            G = r + self.gamma * G
            if (s, a) in self.Q:
                self.Q[(s, a)] = self.Q[(s, a)] + self.lr * (G - self.Q[(s, a)]) # update rule
            else:
                self.Q[(s, a)] = G # initialize to the first example
    
  
    def draw_plots(self, once=False):

        f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        while True:
            try:
                d = {}
                i = 0
                for sa, G in self.Q.items():
                    
                    state = sa[0]
                    d[i] = {"y_difference":state[0],
                            "player_vel":state[1],
                            "next_pipe_dist_to_player":state[2],
                            "pipe_next_pipe_difference":state[3],
                            "action":self.get_argmax_a(state),
                            "return":G,
                            "count_seen":self.s_a_counts[sa]}
                    i += 1

                df = pd.DataFrame.from_dict(d, "index")
                df = df.pivot_table(index="y_difference", columns="next_pipe_dist_to_player", values=["action", "return", "count_seen"])

                ax1.cla()
                ax2.cla()
                ax3.cla()
                ax4.cla()
                self.plot_learning_curve(ax1)
                self.plot_actions(ax2, df)
                self.plot_expected_returns(ax3, df)
                self.plot_states_seen(ax4, df)
                
                if once:
                    plt.show()
                    break
                else:
                    plt.pause(5)
            except:
                pass


name = "state_aggregated_monte_carlo"
agent = StateAggregatedMonteCarlo(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
