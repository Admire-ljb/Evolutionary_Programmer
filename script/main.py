from operatorlib.population import *
from operatorlib.initialize import *
from operatorlib.decoder import *
import time
from figplot import *


def fit(fitness_wight_factor, constraint, time_used, e_t, lambda_1, lambda_2):
    return fitness_wight_factor * lambda_1 + lambda_2 * time_used / e_t + np.sum(constraint > 0)


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
            t0 = time.time()
            temp.evolve(print_=0)
            t = time.time() - t0
            temp_fit = fit(temp.individuals[0].fitness_wight_factor, temp.individuals[0].constraint, t, self.e_t, 0.4, 0.6)
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
            cnt += 1
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
    p_.evolve(print_=1)
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


if __name__ == "__main__":
    global_map = test_map()
    generate_map_in_constrain(global_map, 5, 5, 5)
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
    trajectories = []
    genomes = rd_genomes(10, genome_a)
    start = np.array([10, 10, global_map.terrain.map(10, 10)])
    goal = np.array([100, 70, global_map.terrain.map(100, 70)])
    p_1 = test_population(start, goal, global_map, genome_a)
    new_trajectory(trajectories, p_1.individuals[0].trajectory, "origin")

    # soft_cluster = SoftwareCluster(genomes, start, goal, global_map, e_t=3, time_limit=60)
    # soft_cluster.evolve(print_=1)
    # p_2 = test_population(start, goal, global_map, soft_cluster.genomes[0])
    # new_trajectory(trajectories, p_2.individuals[0].trajectory, 'evolved')
    # PLOT
    figure = plt.figure()
    ax_ = Axes3D(figure)
    plt_terrain(start, goal, global_map, ax_)
    plt_3d_trajectories(ax_, trajectories)
    plt_contour(start, goal, global_map, trajectories)

    # fig = plt.figure()
    # plt.plot([p.individuals[i].fitness_wight_factor for i in range(len(p.data))])