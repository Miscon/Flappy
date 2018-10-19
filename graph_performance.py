import pickle
import matplotlib.pyplot as plt
import numpy as np


class FlappyAgent:
    def __init__(self):
        self.Q = {}
        self.greedy_policy = {}
        self.episode = []

        self.episode_count = 0
        self.frame_count = 0
        self.score = 0
        self.episodes_scores_frames = []
        return


agent = FlappyAgent()
with open("on_policy_monte_carlo.pkl", "rb") as f:
    agent = pickle.load(f)
    print(agent.episode_count)


    
fig = plt.figure()
ax = plt.axes()

total_episodes = 100000

x_base = [a[0] for a in agent.episodes_scores_frames[:total_episodes]]
y_base = [a[1] for a in agent.episodes_scores_frames[:total_episodes]]

x = []
y = []
size = len(agent.episodes_scores_frames[:total_episodes])
interval = 2000
for i in range(size/interval):
    start = interval*i
    end = (interval*(i+1))-1
    x.append(x_base[end])
    y.append(sum(y_base[start:end]) / float(interval))

print(x)





print(len(x))

print(len(y))
ax.plot(x, y);

plt.show()