from exploitation_exploration import *


def evolve(g_map, start, goal, genome_64_bits):
    n_p = decode_n_p(genome_64_bits[2:4])
    swarms = []
    for p_i in range(n_p):
        swarms.append(Population(start, goal, g_map, _num_individual=genome_64_bits[5:8],
                                 curve_type=genome_64_bits[9:11], _cp=genome_64_bits[0:2],
                                 _rank_param=genome_64_bits[18:20], _elitism=genome_64_bits[16:18]))
    end_condition = end_criterion(_end=genome_64_bits[36:37], _end_param=genome_64_bits[37:40])
    result = []
    for each_p in swarms:
        each_p.gen_max = end_condition[1]
        sort(each_p, genome_64_bits[11:13], genome_64_bits[13:16])
        # TODO time
        fit_evolve = []
        while not determine_terminate(end_condition, each_p):
            if each_p.elitism > 0:
                children = each_p.data[0: int(each_p.elitism * each_p.num_individual) +1]
            else:
                children = []
            while len(children) < each_p.num_individual:
                parents = [selection(each_p, _se=genome_64_bits[20:22], _se_param=genome_64_bits[22:24]),
                           selection(each_p, _se=genome_64_bits[20:22], _se_param=genome_64_bits[22:24])]
                offsprings = exploitation(each_p, parents, _exploit=genome_64_bits[24:27],
                                          _exploit_param=genome_64_bits[27:29], _twins=genome_64_bits[29:30])
                mutants = exploration(each_p, offsprings, _explore=genome_64_bits[30:33],
                                      _explore_param=genome_64_bits[33:35], _infer=genome_64_bits[35:36])
                children += mutants
            each_p.data = children[0:each_p.num_individual]
            each_p.generation += 1
            each_p.explore_param = None
            sort(each_p, genome_64_bits[11:13], genome_64_bits[13:16])

            fit_evolve.append(each_p.data[0].fitness_wight_factor)
        result.append(each_p.data[0])
        plt.plot(fit_evolve)
        # plt.plot(penalty_evolve)

    return result[0]


def end_criterion(_end, _end_param):
    dict_end_param = {'000': 20,
                      '001': 40,
                      '010': 60,
                      '011': 80,
                      '100': 100,
                      '101': 150,
                      '110': 200,
                      '111': 500}
    if not int(_end):
        key_words = 'gen'
    else:
        key_words = 'time'
    return [key_words, dict_end_param[_end_param]]


def determine_terminate(end_condition, population):
    if end_condition[0] == 'gen':
        bool_value = end_condition[1] == population.generation
    else:
        bool_value = 1
        # TODO time
    return bool_value


def plt_terrain(start_point, target_point, g_map, ax_3d):
    if start_point[0] < target_point[0]:
        begin_x = int(start_point[0])
        end_x = int(target_point[0])
    else:
        begin_x = int(target_point[0])
        end_x = int(start_point[0])
    if start_point[1] < target_point[1]:
        begin_y = int(start_point[1])
        end_y = int(target_point[1])
    else:
        begin_y = int(target_point[1])
        end_y = int(start_point[1])
    x = np.arange(begin_x - 10, end_x + 10, 1)
    y = np.arange(begin_y - 10, end_y + 10, 1)
    x, y = np.meshgrid(x, y)
    x = x.reshape(1, -1)[0]
    y = y.reshape(1, -1)[0]
    z = np.array([g_map.terrain.map(x[cnt], y[cnt]) for cnt in range(len(x))])
    ax_3d.scatter3D(x, y, z, color='#00CED1')


def map2z_point(arr):
    list_arr = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            list_arr.append(arr[i][j][2])

    list_arr = np.array(list_arr).reshape(arr.shape[0], arr.shape[1])
    return list_arr


if __name__ == "__main__":
    temp = terrain.generate_map()[0]
    terr = Terr(temp)
    missile, radar, nfz = mapconstraint.generate_constraint(3, 3, 0, terr.points)
    global_map = Map(terr, missile, radar, nfz)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for data in p.data:
    #     plt_fig(data.trajectory, ax)
    # fig = plt.figure()
    # plt.plot([p.data[i].fitness_wight_factor for i in range(len(p.data))])
    genome_64_bits = '0000001000' \
                     '0000101000' \
                     '0000000100' \
                     '0000000011' \
                     '0000000000' \
                     '0000000000' \
                     '0000'
    start = np.array([0, 0, global_map.terrain.map(0, 0)])
    goal = np.array([50, 50, global_map.terrain.map(50, 50)])
    result = evolve(global_map, start, goal, genome_64_bits)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt_fig(result.trajectory, ax)
    plt_terrain(start, goal, global_map, ax)