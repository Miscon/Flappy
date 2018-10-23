#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

agent_type = sys.argv[1]


plt.ion()
while True:
    df = pd.read_csv("{}/scores.csv".format(agent_type))

    plt.plot(df["frame_count"], df["avg_score"])
    plt.fill_between(df["frame_count"], df["interval_upper"], df["interval_lower"], alpha=0.5)

    plt.xlabel("Number of frames")
    plt.ylabel("Average score")
    plt.title("{}\nAverage score from 50 games after training on n frames".format(agent_type.replace("_", " ")))
    
    plt.draw()
    plt.pause(20)
    plt.clf()


