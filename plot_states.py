#!/usr/bin/python

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys


class FlappyAgent:
    def __init__(self):
        self.Q = {}
        self.greedy_policy = {}
        self.s_a_counts = {} # for graphs
        self.episode = []
  
        self.episode_count = 0 # for graphs
        self.frame_count = 0 # for graphs
        return

    def get_state_argmax_a(self, s):

        G0 = self.Q.get((s, 0))
        G1 = self.Q.get((s, 1))

        if G0 == None and G1 == None:
            return None
        elif G0 == None:
            return 0
        elif G1 == None:
            return 1
        else:
            if G0 > G1:
                return 0
            else:
                return 1

def plot_actions(data):
    
    df = data.pivot_table(index="y_difference", columns="next_pipe_dist_to_player", values="action")

    fig = plt.figure()
    ax = fig.gca()
    ax = sns.heatmap(df, annot=False, cbar=True)
    plt.xticks(rotation='horizontal')
    plt.yticks(rotation='horizontal')
    ax.set_title("Best action")
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

def plot_expected_return(data):
    df = data.pivot_table(index="y_difference", columns="next_pipe_dist_to_player", values="return")

    fig = plt.figure()
    ax = fig.gca()
    ax = sns.heatmap(df, annot=False, cbar=True)
    plt.xticks(rotation='horizontal')
    plt.yticks(rotation='horizontal')
    ax.set_title("Expected return")
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

def plot_states_seen(data):
    # TODO the pivot uses averages of seen, need to sum
    df = data.groupby(["y_difference", "next_pipe_dist_to_player"]).agg({"count_seen": np.sum})
    df = df.pivot_table(index="y_difference", columns="next_pipe_dist_to_player", values="count_seen")

    fig = plt.figure()
    ax = fig.gca()
    ax = sns.heatmap(df, annot=False, cbar=True)
    plt.xticks(rotation='horizontal')
    plt.yticks(rotation='horizontal')
    ax.set_title("Count states seen")
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()


agent_type = sys.argv[1]
agent = FlappyAgent()

with open("{}/newest.pkl".format(agent_type), "rb") as f:
    agent = pickle.load(f)
    print("Loading snapshot {}".format(agent.episode_count))

df = pd.DataFrame()

for sa, G in agent.Q.items():
    state = sa[0]
    df = df.append({"player_y":state[0],
                    "player_vel":state[1],
                    "next_pipe_dist_to_player":state[2],
                    "next_pipe_top_y":state[3],
                    "y_difference":state[0] - state[3],
                    "action":agent.get_state_argmax_a(state),
                    "return":G,
                    "count_seen":agent.s_a_counts[sa]}, ignore_index=True)



plot_actions(df)
plot_expected_return(df)
plot_states_seen(df)
