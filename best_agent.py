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


class BestAgent(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
  
    
    def map_state(self, state):

        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]
        next_pipe_top_y = state["next_next_pipe_top_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]
        next_distance_to_pipe = state["next_pipe_dist_to_player"]

        player_pipe_difference = player_y - pipe_top_y
        pipe_next_pipe_difference = pipe_top_y - next_pipe_top_y


        # [<0, 0 to 100 in 10 bins, >100]
        step = math.ceil(100/5)
        bins = np.concatenate(([-235], np.arange(0, 100, step), [235]), axis=0)
        player_pipe_difference_bin = np.digitize([player_pipe_difference], bins)[0]

        # [-8, -7 to 9, -10]
        player_vel_bin = (0 if player_vel == -8 else (2 if player_vel == 10 else 1))

        # [<-75, -75 to 0, 0 to 75, >75]
        step = math.ceil((157+157)/4)
        bins = np.arange(-157, 157, step)
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], bins)[0]

        # [0 to 100 in 10 bins, 100 to 144, >144]   maybe change 144 to 164
        step = math.ceil((100)/5)
        bins = np.concatenate((np.arange(0, 100, step), [145, 310]), axis=0)
        distance_to_pipe_bin = np.digitize([distance_to_pipe], bins)[0]
        
                      
        return (player_pipe_difference_bin, player_vel_bin, pipe_next_pipe_difference_bin, distance_to_pipe_bin)

    
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
            Qs1a = self.initial_return_value

        # Calculate return
        G = r + self.gamma * self.get_max_a(s2) 

        # Update Q table
        self.Q[(s1, a)] = Qs1a + self.lr * (G - Qs1a) # update rule
    

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
                            "next_pipe_dist_to_player":state[3],
                            "pipe_next_pipe_difference":state[2],
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


name = "best_agent"
agent = BestAgent(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
