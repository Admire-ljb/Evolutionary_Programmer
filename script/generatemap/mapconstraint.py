from numpy import random
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
        s = 0
        try:
            s = self.points[index, 2]
        except IndexError:
            pass
        return s


class Map:
    def __init__(self, terr, missile, radar, nfz):
        self.terrain = terr
        self.missile = missile
        self.radar = radar
        self.nfz = nfz


class Missile:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class Radar:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class NFZ:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


def generate_missile_or_radar(terrain, dtype="missile"):
    center = terrain[random.randint(len(terrain))]
    radius = random.rand() * 20 + 10
    if dtype == "missile":
        obj = Missile(center, radius)
    else:
        obj = Radar(center, radius)
    return obj


def generate_nfz(terrain):
    x_min = random.rand() * (terrain[-1][0]-30)
    x_max = x_min + random.rand()*20 + 10
    y_min = random.rand() * (terrain[-1][1]-30)
    y_max = y_min + random.rand()*20 + 10
    nfz = NFZ(x_min, x_max, y_min, y_max)
    return nfz


def generate_constraint(num_of_missile, num_of_radar, num_of_nfz, terr):
    missile = []
    radar = []
    nfz = []
    for i in range(0, num_of_missile):
        missile.append(generate_missile_or_radar(terr))
    for i in range(0, num_of_radar):
        radar.append(generate_missile_or_radar(terr, "radar"))
    for i in range(0, num_of_nfz):
        nfz.append(generate_nfz(terr))
    return missile, radar, nfz


