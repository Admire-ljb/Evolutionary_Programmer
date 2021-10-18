import numpy as np

from operatorlib.population import *
from operatorlib.initialize import *
from operatorlib.decoder import *
import time
from figplot import *
import tf
import matplotlib.pyplot as plt


algorithms_lib = {
    'GA':        '0100001001'
                 '1001011101'
                 '1100001101'
                 '0111101110'
                 '0111011011'
                 '0100101000'
                 '0010',
    'CIPSO':     '0100011000'
                 '0111101010'
                 '1010011101'
                 '1000000110'
                 '0000000000'
                 '0000000000'
                 '0000',
    'HHPSO':     '0100001001'
                 '1001011101'
                 '1100001101'
                 '0111101110'
                 '0111011011'
                 '0100101000'
                 '0010',
    'JADE':      '0111011100'
                 '0111111010'
                 '0101111111'
                 '0010100110'
                 '1000000101'
                 '0111101111'
                 '0000',
    'CIPDE':     '0001100111'
                 '1101110111'
                 '0110110000'
                 '1101011110'
                 '0001011010'
                 '1010110010'
                 '1011',

    'mWPS':      '0000110101'
                 '1110100111'
                 '1010100000'
                 '1011010110'
                 '1100000000'
                 '0101000010'
                 '0010',
    'HSGWO-SOS': '0011011011'
                 '0110010000'
                 '0001101111'
                 '1100100110'
                 '0100011010'
                 '0110000100'
                 '0001',


}


def fit(fitness_wight_factor, constraint, time_used, e_t, lambda_1, lambda_2):
    return fitness_wight_factor * lambda_1 + lambda_2 * time_used / e_t + np.sum(constraint > 0)


def compare_average(planner):
    fit_sum = 0
    time_sum = 0
    suc_time = 0
    times = 100
    for i in range(times):
        temp = Population(planner.start, planner.goal, planner.global_map, planner.soft_inf)
        a = time.time()
        temp.evolve()
        time_sum += time.time() - a
        fit_sum += temp.individuals[0].fitness_wight_factor
        if np.sum(temp.individuals[0].constraint) < 2:
            suc_time += 1

    return suc_time/ times, fit_sum / times, time_sum / times


class SoftwareCluster:
    def __init__(self, random_genomes, start, end, g_map, e_t, time_limit):
        self.e_t = e_t
        self.start = start
        self.end = end
        self.map = g_map
        self.num_genomes = len(random_genomes)
        self.genomes = np.array(random_genomes)
        self.fit = np.array([])
        self.time_limit = time_limit
        self.update_generation(random_genomes)
        self.sort()

    def sort(self):
        index = np.argsort(self.fit)
        self.fit = self.fit[index]
        self.genomes = self.genomes[index]

    def selection(self):
        total = np.sum(1/self.fit)
        prob = (1 / self.fit) / total
        for i in range(1, self.genomes.size):
            prob[i] += prob[i-1]
        rand_0_1 = np.random.rand(self.genomes.size*2, 1)
        selection_basis = np.tile(prob, self.genomes.size*2).reshape(self.genomes.size*2, -1)
        return self.genomes[np.argmax(selection_basis >= rand_0_1, axis=1)]

    def update_generation(self, variants, data_pre=np.array([])):
        fit_tmp = []
        genomes_tmp = []
        cnt = 0
        for each in data_pre:
            genomes_tmp.append(each)
            fit_tmp.append(self.fit[cnt])
            cnt += 1
        for each in variants:
            temp = Population(self.start, self.end, self.map, SoftInformation(each))
            temp_fit = 0
            for avr in range(3):
                t0 = time.time()
                temp.evolve(print_=0)
                t = time.time() - t0
                temp_fit += fit(temp.individuals[0].fitness_wight_factor, temp.individuals[0].constraint, t, self.e_t, 0.4, 0.6)
            temp_fit /= 3
            genomes_tmp.append(each)
            fit_tmp.append(temp_fit)
        self.genomes = np.array(genomes_tmp)
        self.fit = np.array(fit_tmp)
        self.sort()
        self.genomes = self.genomes[0:self.num_genomes]
        self.fit = self.fit[0:self.num_genomes]

    def evolve(self, print_=0, elitism=0.1):
        st = time.time()
        cnt = 0
        tmp = []
        while time.time() - st < self.time_limit:
            data_pre = self.genomes[0:int(elitism * self.genomes.size) + 1]
            if print_ == 1:
                print("===============", cnt, "th evolve===============")
                print("fit:", self.fit[0])
                tmp.append(self.fit[0])

            off_spring = crossover(self.selection())
            variants = mutation(off_spring)
            self.update_generation(variants, data_pre)
            # trs = []
            # count = 1
            # for each in self.genomes:
            #     p_ = test_population(self.start, self.end, self.map, each)
            #     trs.append(Trajectory(p_.individuals[0].trajectory, line_types[len(trs)], str(cnt) + '_' + str(count)))
            #     count += 1
            # cnt += 1
            # plt_contour(start, goal, global_map, trs)
        # x_ = np.linspace(0, cnt, 1)
        # y_ = np.array(tmp)
        # plt.figure()
        # plt.plot(x_, y_, color="red", linewidth=1)
        # plt.show()


def crossover(parents):
    children = []
    for x in range(0, parents.size, 2):
        point_ = np.random.randint(0, 64, 4)
        point_.sort()
        child = parents[x][0:point_[1]] + parents[x+1][point_[1]:point_[2]]\
                + parents[x][point_[2]:point_[3]] + parents[x+1][point_[3]::]
        children.append(child)
    return np.array(children)


def mutation(children, pm_=0.01):
    r_ = np.random.rand(64, children.size)
    inx = np.where(r_ < pm_)
    cnt = 0
    while cnt < inx[0].size:
        child = children[inx[1][cnt]]
        pointer = inx[0][cnt]
        pre = child[0:pointer]
        post = child[pointer+1::]
        cur = str(1 - int(child[pointer]))
        children[inx[1][cnt]] = pre+cur+post
        cnt += 1
    return children


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
    temps = rd_genomes(genome, 10)
    cluster = SoftwareCluster(temps, st, end, g_map, 5)
    cluster.evolve()


def test_population(start_, goal_, g_map, genome):
    p_ = Population(start_, goal_, g_map, SoftInformation(genome))
    # p = Population(start, goal, global_map, SoftInformation(genomes[0]))
    p_.evolve(print_=0)
    return p_


def save_data(file_name, data_name):
    import pickle
    output = open(file_name, 'wb')
    pickle.dump(data_name, output)
    output.close()


def load_data(file_name):
    # import pprint
    import pickle
    pkl_file = open(file_name, 'rb')
    data_name = pickle.load(pkl_file)
    # pprint.pprint(data_name)
    pkl_file.close()
    return data_name

def random_test():
    # global_map = test_map()
    # global_map = load_data('data/random_')
    p_2 = load_data('data/random_1_evolved_population')
    global_map = p_2.global_map

    # global_map = load_data('data/lot_map1')
    # generate_map_in_constrain(global_map, 20, 20, 20)
    # genome_a = '0000011000' \
    #            '0000000000' \
    #            '0000000000' \
    #            '0000000101' \
    #            '0000000000' \
    #            '0000000000' \
    #            '0000'
    genome_a = '0100001001' \
               '1001011101'\
               '1100001101'\
               '0111101110'\
               '0111011011'\
               '0100101000'\
               '0010'
    # genome_b = '0011010011001001010111110101111101010011101010101101011010001010'
    # start = np.array([0, 0, global_map.terrain.map(0, 0)])
    # goal = np.array([50, 50, global_map.terrain.map(50, 50)])
    # p = Population(start, goal, global_map, SoftInformation(genome_b))
    # p.evolve(print_=1)

    genomes = rd_genomes(10, genome_a)
    # Performance
    start = np.array([10, 10, global_map.terrain.map(10, 10)])
    goal = np.array([100, 70, global_map.terrain.map(100, 70)])
    p_2 = Population(start, goal, global_map, p_2.soft_inf)
    p_1 = []
    for each in algorithms_lib:
        p_1.append([Population(start, goal, global_map, SoftInformation(algorithms_lib[each])), each])
        p_1[-1][0].evolve()
    p_2.evolve()
    trajectories = []
    fig, axs = plt.subplots(2, 1)
    for each in p_1:
        new_trajectory(trajectories, each[0].individuals[0].trajectory, each[1])
        t = np.arange(0, each[0].gen_max, 1)
        axs[0].plot(t, each[0].fitness_history, trajectories[-1].linestyle, label=each[1])
        axs[1].plot(t, each[0].constraint_history, trajectories[-1].linestyle, label=each[1])
        # ax.plot(t, s)
    new_trajectory(trajectories, p_2.individuals[0].trajectory, 'evolved')
    trajectories[-1].linestyle = trajectories[-1].linestyle[0:-1]
    t = np.arange(0, p_2.gen_max, 1)
    axs[0].plot(t, p_2.fitness_history, trajectories[-1].linestyle, color='orange', label='evolved')
    axs[1].plot(t, p_2.constraint_history, trajectories[-1].linestyle,  color='orange', label='evolved')
    axs[0].set_ylabel('fitness')
    axs[1].set_ylabel('constraint')
    axs[1].set_xlabel('generation')
    axs[1].set_ylim(ymin=0)
    # axs[1].set_xlim(xmin=0)
    # axs[0].set_ylim(ymin=0)
    # axs[0].set_xlim(xmin=0)
    # axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #               ncol=4, mode="expand", borderaxespad=0.)
    plt.show()

    plt_contour(start, goal, global_map, trajectories)

    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    # ax.grid()

    # adaptive verification
    soft_cluster = SoftwareCluster(genomes, start, goal, global_map, e_t=3, time_limit=40000)
    soft_cluster.evolve(print_=0)
    # p_1 = test_population(start, goal, global_map, genome_a)
    # p_2 = test_population(start, goal, global_map, soft_cluster.genomes[0])
    # trajectories = []
    # new_trajectory(trajectories, p_1.individuals[0].trajectory, "origin")
    # new_trajectory(trajectories, p_2.individuals[0].trajectory, 'evolved')
    # plt_contour(start, goal, global_map, trajectories)
    # PLOT
    # trajectories = []
    figure = plt.figure()
    ax_ = Axes3D(figure)
    plt_terrain(start, goal, global_map, ax_)
    plt_3d_trajectories(ax_, trajectories)
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(5)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    # plt.rcParams.update({'font.size': 70})
    # plt.legend(loc='best')
    # fig = plt.figure()
    # plt.plot([p.individuals[i].fitness_wight_factor for i in range(len(p.data))])


if __name__ == "__main__":
    p_2 = load_data('data/hug_1_origin_population')
    p_1 = load_data('data/hug_1_evolved_population')
    global_map = p_2.global_map
    # start = np.array([40, 10, global_map.terrain.map(40, 10)])
    # goal = np.array([60, 70, global_map.terrain.map(60, 70)])
    # start = np.array([110, 10, global_map.terrain.map(110, 10)])
    # goal = np.array([10, 70, global_map.terrain.map(10, 70)])
    start = np.array([10, 35, global_map.terrain.map(10, 35)])
    goal = np.array([100, 35, global_map.terrain.map(100, 35)])
    p_2 = Population(start, goal, global_map, p_2.soft_inf)
    p_1 = Population(start, goal, global_map, p_1.soft_inf)
    p_2.evolve()
    p_1.evolve()
    trajectories = []
    new_trajectory(trajectories, p_1.individuals[0].trajectory, "origin_")
    new_trajectory(trajectories, p_2.individuals[0].trajectory, 'evolved_')
    plt_contour(np.array([10, 10, 0]), np.array([110, 90, 0]), global_map, trajectories)
    trajectories = []
    #
    for i in range(1, 5):
        p_1 = load_data('data/points4_'+str(i)+'_p_1')
        p_2 = load_data('data/points4_'+str(i)+'_p_2')
        new_trajectory(trajectories, p_1.individuals[0].trajectory, "origin_"+str(i))
        new_trajectory(trajectories, p_2.individuals[0].trajectory, 'evolved_'+str(i))
        trajectories[-1].linestyle = trajectories[-1].linestyle[0:2] + 'r'
        trajectories[-2].linestyle = trajectories[-2].linestyle[0:2] + 'b'
    plt_contour(np.array([10, 10, 0]), np.array([110, 90, 0]), global_map, trajectories)
    # for i in range(1, 5):
    #     p_1 = load_data('data/points_'+str(i)+'_p_1')
    #     p_2 = load_data('data/points_'+str(i)+'_p_2')
    #     temp = []
    #     new_trajectory(temp, p_1.individuals[0].trajectory, "origin_"+str(i))
    #     new_trajectory(temp, p_2.individuals[0].trajectory, 'evolved_'+str(i))
    #     temp[0].linestyle = sq[i-1] + temp[0].linestyle[1:3]
    #     temp[1].linestyle = sq[i-1] + temp[1].linestyle[1:3]
    #     trajectories.append(temp[0])
    #     trajectories.append(temp[1])
    # plt_contour(np.array([10, 10, 0]), np.array([100, 70, 0]), global_map, trajectories)
    # figure = plt.figure()
    # ax_ = Axes3D(figure)
    # plt_terrain(np.array([10, 10, 0]), np.array([100, 70, 0]), global_map, ax_)
    # plt_3d_trajectories(ax_, trajectories)
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(5)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})