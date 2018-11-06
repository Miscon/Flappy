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


class ReDiscretize(FlappyAgent):
    def __init__(self, name, x=10, y=10):
        FlappyAgent.__init__(self, name)

        x += 1 # the split is off by one
        y += 1
        self.gamma = 0

        self.last_10 = []

        self.player_pipe_difference_bins = np.linspace(13, 76, y)
        self.player_vel_bins = [-8, -7, 0, 5, 10]
        self.pipe_next_pipe_difference_bins = np.linspace(-158, 158, 5)
        self.distance_to_pipe_bins = [0, 65, 100] #[b-1 for b in np.geomspace(1, 66, num=x)]
        
    
    def map_state(self, state):

        player_y = state["player_y"]
        player_vel = state["player_vel"]
        pipe_top_y = state["next_pipe_top_y"]
        next_pipe_top_y = state["next_next_pipe_top_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]
        next_distance_to_pipe = state["next_next_pipe_dist_to_player"]

        player_pipe_difference = player_y - pipe_top_y
        pipe_next_pipe_difference = pipe_top_y - next_pipe_top_y


        player_pipe_difference_bin = np.digitize([player_pipe_difference], self.player_pipe_difference_bins)[0]
        player_vel_bin = np.digitize([player_vel], self.player_vel_bins)[0]
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], self.pipe_next_pipe_difference_bins)[0]
        distance_to_pipe_bin = np.digitize([distance_to_pipe], self.distance_to_pipe_bins)[0]


        self.last_10.append(((player_pipe_difference, player_vel, pipe_next_pipe_difference, distance_to_pipe), (player_pipe_difference_bin, player_vel_bin, pipe_next_pipe_difference_bin, distance_to_pipe_bin)))
        if len(self.last_10) > 10:
            self.last_10.pop(0)

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
            Qs1a = self.initial_q_value(s1, a)

        # Calculate return
        # G = r
        # if r != self.reward_values()["positive"] and r != self.reward_values()["loss"]:
        G = r + self.gamma * self.get_max_a(s2) 

        # lr = 1.0 / self.s_a_counts[(s1,a)]

        # Update Q table
        self.Q[(s1, a)] = Qs1a + self.lr * (G - Qs1a) # update rule
    

    def initial_q_value(self, s, a):
        return 0
        neg = -100
        pos = 10
        
        if s[0] == 0:
            if a == 0:
                return neg
            else:
                return pos
        else:
            if a == 0:
                return pos
            else:
                return neg
        

    def get_max_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.initial_q_value(state, 0)
        if G1 is None:
            G1 = self.initial_q_value(state, 1)

        if G0 > G1:
            return G0
        else:
            return G1


    def get_argmax_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.initial_q_value(state, 0)
        if G1 is None:
            G1 = self.initial_q_value(state, 1)

        if G0 == G1:
            return random.randint(0, 1)
        elif G0 > G1:
            return 0
        else:
            return 1

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


name = "re_discretize"
x = 10
y = 10

try:
    name += "_{}_{}".format(sys.argv[2], sys.argv[3])
    x = sys.argv[2]
    y = sys.argv[3]
except:
    pass

agent = ReDiscretize(name, int(x), int(y))

try:
    with open("{}/agent.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
