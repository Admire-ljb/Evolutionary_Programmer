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
        self.generation = 0
        self.sort_cache = 0
        self.selection_cache = 0
        self.data = np.array([initialize(g_map, self.num_cp, start, self.goal_r)
                                            for x in range(self.num_individual)])
        self.velocity = np.zeros(self.data.shape)
        self.p_best = self.data.copy()
        self.individuals = np.array([Individual(self, self.data[x]) for x in range(self.num_individual)])
        self.rank_probability = None
        self.exploitation_params = None
        # self.sort_basis = None

    def update_generation(self, data):
        self.generation += 1
        self.sort_cache = 0
        self.selection_cache = 0
        self.data = data
        individuals = instantiation(self, self.data)
        if self.soft_inf.exploit == "pso_exploit":
            for x in range(individuals.size):
                if self.sort(np.array([individuals[x], self.individuals[x]]))[0] == 0:
                    self.p_best[x] = self.data[x].copy()
        self.individuals = individuals

    def sort(self, sort_data):
        return eval(self.soft_inf.so)(self, sort_data, self.soft_inf.so_param)

    def selection(self, select_pool, num_of_results):
        return eval(self.soft_inf.se)(self, select_pool, self.soft_inf.se_param, num_of_results)

    def exploitation(self, individuals):
        return eval(self.soft_inf.exploit)(self, individuals, self.soft_inf.exploit_param)


def test_population(g_map):
    # t0 = time.clock()
    genomes = '1000001100' \
              '0110101000' \
              '0111101001' \
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
    inx = [p.sort(p.individuals)]
    p.individuals = p.individuals[inx]
    p.data = p.data[inx]
    # p.update_generation()
    b = p.selection(p.individuals, p.num_individual)
    figure = plt.figure()
    ax = Axes3D(figure)

    plt_fig(p.individuals[b[0]].trajectory, ax, "parent_1")
    plt_fig(p.individuals[b[1]].trajectory, ax, "parent_2")
    plt_fig(p.individuals[0].trajectory, ax, "parent_3")
    # c = p.selection(p.data)
    d = p.exploitation(b)
    p.update_generation(d)
    plt_fig(p.individuals[0].trajectory, ax, "child_1")
    plt_fig(p.individuals[1].trajectory, ax, "child_2")
    # for data in p.data:
    #     plt_fig(data.trajectory, ax)
    # terrain.plt_terrain(p.start, p.goal, p.global_map, ax)
    # fig = plt.figure()
    # plt.plot([p.data[i].fitness_wight_factor for i in range(len(p.data))])