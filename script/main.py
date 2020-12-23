from generatemap import terrain
from generatemap import mapconstraint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Map:
    def __init__(self, terr, missile, radar, nfz):
        self.Terrain = terr
        self.missile = missile
        self.radar = radar
        self.nfz = nfz


if __name__ == "__main__":
    terr = terrain.generate_map()[0]
    missile, radar, nfz = mapconstraint.generate_constraint(2, 3, 2, terr)
    global_map = Map(terr, missile, radar, nfz)

    """visualize"""
    fig = plt.figure()
    terrain.plt_fig(terr, fig)
    # terrain.constraint_plt(missile, radar, nfz, fig)
    plt.show()