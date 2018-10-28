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

class StateAggregatedQLearning(FlappyAgent):
    
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
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

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

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


    def plot_learning_curve(self, ax):
        
        df = pd.read_csv("{}/scores.csv".format(self.name))

        ax.plot(df["frame_count"], df["avg_score"])
        ax.fill_between(df["frame_count"], df["interval_upper"], df["interval_lower"], alpha=0.5)

        ax.set_xlabel("Number of frames")
        ax.set_ylabel("Average score")
        ax.set_title("{}\nAverage score from 10 games".format(self.name.replace("_", " ")))


name = "state_aggregated_q_learning"
agent = StateAggregatedQLearning(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
