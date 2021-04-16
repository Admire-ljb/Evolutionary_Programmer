from operatorlib.population import *
from operatorlib.initialize import *
from operatorlib.decoder import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


class SoftwareCluster:
    def __init__(self, random_genomes, start, end, g_map, e_t):
        self.e_t = e_t
        self.genomes = random_genomes
        self.software_cluster = []
        self.fit = np.zeros(len(random_genomes))
        for each in random_genomes:
            temp = Population(start, end, g_map, SoftInformation(each))
            t0 = time.clock()
            temp.evolve(print_=0)
            t = time.clock() - t0
            self.software_cluster.append((temp, t))

    def fit(self):
        self.fit = np.array([self.software_cluster[i][0].individuals[0].fitness_wight_factor * 0.4 + 0.6 * self.software_cluster[i][1]/ self.e_t
                             for i in range(len(self.genomes))])

    def evolve(self):
        pass


def rd_genomes(mounts, genome=None):
    if genome:
        temp = np.random.randint(0, 2, (mounts - 1, 64)).tolist()
        temp.append(genome)
    else:
        temp = np.random.randint(0, 2, (mounts, 64)).tolist()
    rand_genomes = []
    for each in temp:
        rand_genomes.append(''.join(str(i) for i in each))
    return rand_genomes


def generate_new_genomes(genome, g_map, st, end):
    inf = SoftInformation(genome)
    temps = rd_genomes(genome, 10)
    cluster = SoftwareCluster(temps, st, end, g_map)
    cluster.evolve()


if __name__ == "__main__":
    global_map = test_map(0, 0, 0)
    genome_a = '0000011000' \
               '0000000000' \
               '0000000000' \
               '0000000000' \
               '0000000000' \
               '0000000000' \
               '0000'
    # genome_b = '0011010011001001010111110101111101010011101010101101011010001010'
    # start = np.array([0, 0, global_map.terrain.map(0, 0)])
    # goal = np.array([50, 50, global_map.terrain.map(50, 50)])
    # p = Population(start, goal, global_map, SoftInformation(genome_b))
    # p.evolve(print_=1)
    genomes = rd_genomes(1)
    start = np.array([0, 0, global_map.terrain.map(0, 0)])
    goal = np.array([50, 50, global_map.terrain.map(50, 50)])
    soft_cluster = SoftwareCluster(genomes, start, goal, global_map, 5)
    p = Population(start, goal, global_map, SoftInformation(genomes[0]))
    p.evolve(print_=1)
    figure = plt.figure()
    ax = Axes3D(figure)

    plt_fig(p.individuals[0].trajectory, ax)
    plt_terrain(p.start, p.goal, p.global_map, ax)
    # fig = plt.figure()
    # plt.plot([p.individuals[i].fitness_wight_factor for i in range(len(p.data))])