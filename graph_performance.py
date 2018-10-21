import pickle
import matplotlib.pyplot as plt
import numpy as np


class FlappyAgent:
    def __init__(self):
        self.Q = {} # see picture or rl2, averaging learning rule
        self.greedy_policy = {}
        self.s_a_counts = {} # for graphs
        self.episode = []
  
        self.episode_count = 0 # for graphs
        self.frame_count = 0 # for graphs
        return


agent = FlappyAgent()
with open("on_policy_monte_carlo/scores.pkl", "rb") as f:
    data = pickle.load(f)
    print(agent.episode_count)


    
plt.plot(agent.episode_count)