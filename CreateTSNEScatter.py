import os
import numpy as np
import pandas as pd
import FileHelpers as fh
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--CONTENT_WEIGHT', type=int, default=100)
parser.add_argument('--STYLE_WEIGHT', type=int, default=100)
parser.add_argument('--OUTPUT_PATH', default='//home//michael//git//Wikiart_tSNE//Output//')
args = parser.parse_args()
     
tsne = fh.LoadTSNESpaceAndGenres(args.OUTPUT_PATH, args.CONTENT_WEIGHT, args.STYLE_WEIGHT)
groups = tsne.groupby('Genre')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.X, group.Y, marker='o', alpha=0.5, linestyle='', ms=2, label=name)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
plt.savefig(os.path.join(args.OUTPUT_PATH, "tsne_space_scatterplot.jpg"))
