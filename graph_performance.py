import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mc_df = pd.read_csv("on_policy_monte_carlo/scores.csv")
q_df = pd.read_csv("q_learning/scores.csv")

plt.plot(mc_df["episode_count"], mc_df["avg_score"])
plt.fill_between(mc_df["episode_count"], mc_df["interval_upper"], mc_df["interval_lower"], alpha=0.5)

plt.plot(q_df["episode_count"], q_df["avg_score"])
plt.fill_between(q_df["episode_count"], q_df["interval_upper"], q_df["interval_lower"], alpha=0.5)

plt.show()


plt.plot(mc_df["frame_count"], mc_df["avg_score"])
plt.fill_between(mc_df["frame_count"], mc_df["interval_upper"], mc_df["interval_lower"], alpha=0.5)

plt.plot(q_df["frame_count"], q_df["avg_score"])
plt.fill_between(q_df["frame_count"], q_df["interval_upper"], q_df["interval_lower"], alpha=0.5)

plt.show()