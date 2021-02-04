from initialize import *
from sort_selection import *
from exploitation_exploration import *


class Population:
    """
    Create a number of individuals (i.e. a population).
    """
    def __init__(self, start, goal, g_map, soft_information):
        self.soft_inf = soft_information
        self.start = start
        self.goal = goal
        self.rotation_matrix = rotate_coordinate(start, goal)
        self.goal_r = rotation2st(start, goal, self.rotation_matrix)
        self.distance = euclidean_distance(goal, start)
        self.num_cp = eval(self.soft_inf.num_cp)(self.distance)
        self.num_individual = soft_information.num_individual
        self.global_map = g_map
        self.upper_bound, self.lower_bound = np.array([150, 100]), np.array([0, 0])
        self.len_individual = self.num_cp * 5
        self.gen_max = soft_information.end_param
        # TODO
        self.generation = 31
        self.sort_cache = 0
        self.selection_cache = 0
        self.data = np.array([Individual(self) for x in range(self.num_individual)])
        self.rank_probability = None
        # self.sort_basis = None

    def sort(self, sort_data):
        return eval(self.soft_inf.so)(self, sort_data, self.soft_inf.so_param)

    def selection(self, select_pool):
        return eval(self.soft_inf.se)(self, select_pool, self.soft_inf.se_param)

    def exploitation(self, individuals):
        return eval(self.soft_inf.exploit)(self, individuals, self.soft_inf.exploit_param, self.soft_inf.twins)


def test_population(g_map):
    t0 = time.clock()
    genomes = '1000000000' \
              '0110101000' \
              '1111000100' \
              '0000000010' \
              '0000000000' \
              '0000000000' \
              '0000'
    soft_inf = SoftInformation(genomes)
    pop = Population(np.array([0, 0, g_map.terrain.map(0, 0)]),
                     np.array([50, 50, g_map.terrain.map(50, 50)]), g_map,
                     soft_inf)
    # print(time.clock() - t0)
    # figure = plt.figure()
    # ax = Axes3D(figure)
    # for data in pop.data:
    #     plt_fig(data.trajectory, ax)
    # terrain.plt_terrain(pop.start, pop.goal, pop.global_map, ax)
    # fig = plt.figure()
    # plt.plot([p.data[i].fitness_wight_factor for i in range(len(p.data))])
    return pop


if __name__ == '__main__':
    global_map = test_map(0, 0, 0)
    p = test_population(global_map)
    p.data = p.sort(p.data)
    b = p.selection(p.data)
    c = p.selection(p.data)
    d = p.exploitation([b, c])
    # figure = plt.figure()
    # ax = Axes3D(figure)
    # for data in p.data:
    #     plt_fig(data.trajectory, ax)
    # terrain.plt_terrain(p.start, p.goal, p.global_map, ax)
    # fig = plt.figure()
    # plt.plot([p.data[i].fitness_wight_factor for i in range(len(p.data))])