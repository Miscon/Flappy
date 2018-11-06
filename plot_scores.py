import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys



df = pd.read_csv("scores.txt")
sns.barplot(x="name", y="score", data=df)

plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Agent")
plt.ylabel("Score")
plt.title("Average score from 100 games")
plt.show()