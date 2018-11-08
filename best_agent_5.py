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




class BestAgent5(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)

        self.player_pipe_difference_bins = np.linspace(13, 76, 3)

    
    def map_state(self, state):

        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]

        player_pipe_difference = player_y - pipe_top_y

        player_pipe_difference_bin = np.digitize([player_pipe_difference], self.player_pipe_difference_bins)[0]

        if player_vel <= -8:
            player_vel = 0
        elif player_vel > 0:
            player_vel = 2
        else:
            player_vel = 1

        return (player_pipe_difference_bin, player_vel)

    
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

        # Calculate learning rate
        lr = 1.0 / self.s_a_counts[(s1,a)]

        # Update Q table
        self.Q[(s1, a)] = Qs1a + lr * (G - Qs1a) # update rule


    def draw_plots(self, once=False):

        f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        while True:
            try:
                df = self.make_df()
            
                ax1.cla()
                ax2.cla()
                ax3.cla()
                ax4.cla()
                self.plot_learning_curve(ax1)
                self.plot_actions(ax2, df)
                self.plot_expected_returns(ax3, df)
                self.plot_states_seen(ax4, df)
                ax2.set_xlabel("Agent velocity")
                ax3.set_xlabel("Agent velocity")
                ax4.set_xlabel("Agent velocity")
                
                if once:
                    plt.show()
                    break
                else:
                    plt.pause(5)
            except:
                pass


    def make_df(self):
        d = {}
        i = 0
        for sa, G in self.Q.items():
            
            state = sa[0]
            action = sa[1]
            d[i] = self.state_to_dict(state, action, G)
            i += 1

        df = pd.DataFrame.from_dict(d, "index")
        return df.pivot_table(index="y_difference", columns="player_vel", values=["action", "return", "count_seen"])


    def state_to_dict(self, state, action, G):
        return {"y_difference":state[0],
                "player_vel":state[1],
                "action":self.get_argmax_a(state),
                "return":G,
                "count_seen":self.s_a_counts[(state, action)]}


name = "best_agent_5"

agent = BestAgent5(name)

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
