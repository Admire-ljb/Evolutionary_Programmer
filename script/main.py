from generatemap import terrain
from generatemap import mapconstraint
import matplotlib.pyplot as plt
import numpy as np


class Terr:
    def __init__(self, points):
        self.points = np.array(points)
        temp = points[0][0]
        count = 0
        for i in points:
            if i[0] == temp:
                count += 1
            else:
                break
        self.length = count
        self.width = len(points) // count

    def map(self, x_index, y_index):
        index = x_index * self.length + y_index
        return self.points[index, 2]


class Map:
    def __init__(self, terr, missile, radar, nfz):
        self.terrain = terr
        self.missile = missile
        self.radar = radar
        self.nfz = nfz


if __name__ == "__main__":
    temp = terrain.generate_map()[0]
    terr = Terr(temp)
    missile, radar, nfz = mapconstraint.generate_constraint(2, 3, 2, terr.points)
    global_map = Map(terr, missile, radar, nfz)

    """visualize"""
    fig = plt.figure()
    terrain.plt_fig(terr.points, fig)
    # terrain.constraint_plt(missile, radar, nfz, fig)
    plt.show()