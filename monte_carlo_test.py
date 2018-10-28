from flappy_agent import FlappyAgent

from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st

class OnPolicyMonteCarlo(FlappyAgent):
    def __init__(self, name):
        FlappyAgent.__init__(self, name)
        self.episode = []
  
    
    def map_state(self, state):
        """ We are not using the entire game state as our state as we 
            just want the following values but discretized to at max 15 values
            player_y
            next_pipe_top_y
            next_pipe_dist_to_player
            player_vel
        """

        player_y = int(state['player_y'] / (513.0/15))
        player_vel = int(state['player_vel'] / (20.0/15))
        next_pipe_dist_to_player = int(state['next_pipe_dist_to_player'] / (289.0/15))
        next_pipe_top = int(state['next_pipe_top_y'] / (513.0/15))

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_top)

    
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
        
        lr = 0.1 # learning rate
        gamma = 1 # discount
        
        # instead of append G to returns, use the update rule
        G = 0
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if (s, a) in self.Q:
                self.Q[(s, a)] = self.Q[(s, a)] + lr * (G - self.Q[(s, a)]) # update rule
            else:
                self.Q[(s, a)] = G # initialize to the first example
    
  

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        state = self.map_state(state)

        e = 0.1
        greedy = np.random.choice([False, True], p=[e, 1-e])

        action = self.get_argmax_a(state)
        if not greedy:
            action = random.randint(0, 1)

        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        state = self.map_state(state)
        action = self.get_argmax_a(state)

        return action



name = "on_policy_monte_carlo"
agent = OnPolicyMonteCarlo(name)

try:
    with open("{}/newest.pkl".format(name), "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
except:
    print("Starting new {} agent".format(name))

agent.train()