import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import visualizer
import seaborn as sns



plt.switch_backend('TkAgg')

sns.set()
vis = visualizer.Visualizer()

plt.show()



