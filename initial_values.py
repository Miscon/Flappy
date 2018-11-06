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


class InitialValues(FlappyAgent):
    def __init__(self, name, pos=0, neg=0):
        FlappyAgent.__init__(self, name)

        self.pos = pos
        self.neg = neg

        # Discount results
        self.gamma = 1

        # Discretize results
        x = 3 + 1 # the split is off by one
        y = 10 + 1

        # [<0, 0 to 100 in x bins, >100]
        self.player_pipe_difference_bins = np.concatenate(([-300],  np.linspace(0, 100, y), [500]), axis=0)
        
        # [<-75, -75 to 0, 0 to 75, >75]
        self.pipe_next_pipe_difference_bins = np.linspace(-158, 158, 5)

        # [0 to 100 in x bins, 100 to 164, >164]   maybe change 144 to 164
        # self.distance_to_pipe_bins = np.concatenate((np.linspace(0, 100, x), [145, 310]), axis=0)
        self.distance_to_pipe_bins = [x-1 for x in np.geomspace(1, 311, num=x)]
        
    
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
        # [-8, -7 to 9, -10]
        player_vel_bin = (0 if player_vel == -8 else (3 if player_vel == 10 else (2 if player_vel > 0 else 1)))
        pipe_next_pipe_difference_bin = np.digitize([pipe_next_pipe_difference], self.pipe_next_pipe_difference_bins)[0]
        distance_to_pipe_bin = np.digitize([distance_to_pipe], self.distance_to_pipe_bins)[0]

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
    

    def get_argmax_a(self, state):

        disc_state = self.map_state(state)
        G0 = self.Q.get((disc_state, 0))
        G1 = self.Q.get((disc_state, 1))

        if G0 is None:
            G0 = self.initial_action_return(state, 0)
        if G1 is None:
            G1 = self.initial_action_return(state, 1)

        if G0 == G1:
            return random.randint(0, 1)
        elif G0 > G1:
            return 0
        else:
            return 1


    def initial_action_return(self, state, action):

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

        best_action = 0

        # if player is below the center of the pipe
        difference = player_y - pipe_center_y
        if difference < 0:
            best_action = 1

        # if inside the pipe, do double jumps
        if distance_to_pipe < 100:
            # if player jump last action
            if player_vel == -8:
                best_action = 0
        # else do jump, noop, jump to jump higher
        else:
            # if last action was jump, don't jump right away
            if player_vel == -8:
                best_action = 1

        if action == best_action:
            return self.pos
        else:
            return self.neg


    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        greedy = np.random.choice([False, True], p=[self.e, 1-self.e])

        action = 0
        if greedy:
            action = self.get_argmax_a(state)
        else:
            action = random.randint(0, 1)

        
        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        action = self.get_argmax_a(state)
        return action

    def get_disc_state_argmax_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.initial_return_value
        if G1 is None:
            G1 = self.initial_return_value

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
                            "action":self.get_disc_state_argmax_a(state),
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


name = "initial_values"

pos = 0
neg = 0

try:
    name += "_{}_{}".format(sys.argv[2], sys.argv[3])
    e = sys.argv[2]
except:
    pass

agent = InitialValues(name, int(pos), int(neg))

try:
    with open("{}/agent.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.run(sys.argv[1]) # Use 'train' or 'play'
