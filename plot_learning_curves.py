import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


while True:
    legend = []
    i = 1
    while True:
        try:
            name = sys.argv[i]
            df = pd.read_csv("{}/scores.csv".format(name))
            plt.plot(df["frame_count"], df["avg_score"])
            # plt.fill_between(df["frame_count"], df["interval_upper"], df["interval_lower"], alpha=0.3)
            # plt.fill_between(df["frame_count"], df["max"], df["min"], alpha=0.3)
            legend.append(name)
            i += 1
        except:
            break

    plt.legend(legend, loc="upper left")
    plt.xlabel("Number of frames")
    plt.ylabel("Average score")
    plt.title("Average score from 10 games")
    plt.pause(5)
    plt.clf()