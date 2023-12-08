import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

file_path = "./euro_lang.csv"
data = pd.read_csv(file_path)
data_np = data.values.T

som = MiniSom(4, 4, data_np.shape[1], sigma=0.15, learning_rate=0.5)
som.train(data_np, 70)

languages = data.columns.tolist()
the_grid = GridSpec(4, 4)
for position in range(4 * 4):
    ax = plt.subplot(the_grid[position // 4, position % 4], aspect=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    winning_labels = [
        languages[i]
        for i in range(len(data_np))
        if som.winner(data_np[i]) == (position // 4, position % 4)
    ]
    ax.text(
        0.5,
        0.5,
        "\n".join(winning_labels),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )
    ax.set_facecolor(plt.cm.rainbow(position / 16))

plt.savefig("./image/p1.png")
