from numpy import random


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


