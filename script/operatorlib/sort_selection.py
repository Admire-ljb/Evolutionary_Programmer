from initialize import *

"""old sort, delete"""
# def sort(population, individuals, _so, _so_param):
#     if _so == '00':
#         results = penalty_sort(population, individuals, _so_param)
#     elif _so == '01':
#         results = non_dominated_sort(individuals, _so_param)
#     elif _so == '10':
#         alpha_level_sort(population, individuals, _so_param)
#         results = population.data.sort(key=lambda x: x.sort_basis)
#     else:
#         num_of_un_constraint(population, individuals, _so_param)
#         results = population.data.sort(key=lambda x: x.sort_basis)
#     # population.data.sort(key=lambda x: x.sort_basis)
#     return results


def penalty_sort(population, individuals, _so_param):
    r_k = eval(_so_param)(population.gen_max, population.generation)
    a = time.clock()
    for i in range(100):
        fit = np.array([individuals[i, j].fitness_wight_factor
                        for i in range(individuals.shape[0])
                        for j in range(individuals.shape[1])]).reshape(individuals.shape)
    print(time.clock() - a)
    con = np.array([individuals[i].constraint for i in range(num)])
    penalty = np.sum(con ** 2, axis=1)
    sort_basis = r_k * penalty + fit
    inx = np.argsort(sort_basis, axis=-1, kind='quicksort', order=None)
    sorted_individuals = individuals[inx]
    """Delete"""
    # for individual in population.data:
    #     penalty = sum(individual.constraint ** 2)
    #     individual.sort_basis = individual.fitness_wight_factor + r_k * penalty
    # population.data.sort(key=lambda x: x.sort_basis)
    return sorted_individuals



# def penalty_sort(population, individuals, _so_param):
#     r_k = eval(_so_param)(population.gen_max, population.generation)
#     num = len(individuals)
#     fit = np.array([individuals[i].fitness_wight_factor for i in range(num)])
#     con = np.array([individuals[i].constraint for i in range(num)])
#     penalty = np.sum(con ** 2, axis=1)
#     sort_basis = r_k * penalty + fit
#     inx = np.argsort(sort_basis, axis=-1, kind='quicksort', order=None)
#     sorted_individuals = individuals[inx]
#     """Delete"""
#     # for individual in population.data:
#     #     penalty = sum(individual.constraint ** 2)
#     #     individual.sort_basis = individual.fitness_wight_factor + r_k * penalty
#     # population.data.sort(key=lambda x: x.sort_basis)
#     return sorted_individuals


"""old rk, delete"""
# def penalty_coefficient(gen_max, generation, _so_param):

#     if _so_param == '000':
#         r_k = 1000 / gen_max * generation
#     elif _so_param == '001':
#         r_k = 5000 / gen_max * generation
#     elif _so_param == '010':
#         r_k = 1000 * (1 + np.exp(- generation + gen_max/2)) ** -1 #sigmoid function
#     elif _so_param == '011':
#         r_k = 5000 * (1 + np.exp(- generation + gen_max/2)) ** -1
#     elif _so_param == '100':
#         r_k = 1000 / (gen_max ** 2) * generation ** 2 #ex
#     elif _so_param == '101':
#         r_k = 5000 / (gen_max ** 2) * generation ** 2
#     elif _so_param == '110':
#         r_k = 1000 / (gen_max ** 3) * generation ** 3
#     else:
#         r_k = 5000 / (gen_max ** 3) * generation ** 3
#     return r_k


def rk_linear_1000(gen_max, generation):
    return 1000 / gen_max * generation


def rk_linear_5000(gen_max, generation):
    return 5000 / gen_max * generation


def rk_sigmoid_1000(gen_max, generation):
    return 1000 * (1 + np.exp(- generation + gen_max/2)) ** -1


def rk_sigmoid_5000(gen_max, generation):
    return 5000 * (1 + np.exp(- generation + gen_max/2)) ** -1


def rk_ex2_1000(gen_max, generation):
    return 1000 / (gen_max ** 2) * generation ** 2


def rk_ex2_5000(gen_max, generation):
    return 5000 / (gen_max ** 2) * generation ** 2


def rk_ex3_1000(gen_max, generation):
    return 1000 / (gen_max ** 3) * generation ** 3


def rk_ex3_5000(gen_max, generation):
    return 5000 / (gen_max ** 3) * generation ** 3


def non_dominated_sort(population, individuals, _so_param):
    num = len(individuals)
    values = np.array([individuals[i].fitness for i in range(num)])
    front = fast_non_dominated_sort(values)
    crowding_distance_values = []
    max_min_obj = values.max(axis=0) - values.min(axis=0)
    if np.any(max_min_obj == 0):
        max_min_obj[np.where(max_min_obj == 0)[0]] = 0.01
    all_sort = np.argsort(values, axis=0)
    for i in range(len(front)):
        crowding_distance_values.append(crowding_distance(values, front[i], max_min_obj, all_sort))
    sorted_individuals = np.array([], dtype=np.int32)
    for i in range(len(front)):
        sorted_inx = np.flipud(np.argsort(crowding_distance_values[i]))
        sorted_individuals = np.append(sorted_individuals, individuals[front[i][sorted_inx]])
    """old delete"""
    # for i in range(len(front)):
    #     sorted_inx = np.flipud(np.argsort(crowding_distance_values[i])
    #     front[i] = front[i][sorted_inx]
    #     front_2 = [index_of(front[i][j], front[i]) for j in range(0, len(front[i]))]
    #     front_22 = sort_by_values(front_2[:], crowding_distance_values[i][:])
    #     front_out = [front[i][front_22[j]] for j in range(0, len(front[i]))]
    #     front_out.reverse()
    #     for index in front_out:
    #         sort_basis[index] = count
    #     count += 0.01
    #     count = int(count + 1)
    #     inx = np.argsort(sort_basis, axis=-1, kind='quicksort', order=None)
    #     sorted_individuals = individuals[inx]
    #     count = int(count + 1)
    # inx = np.argsort(sort_basis, axis=-1, kind='quicksort', order=None)
    # sorted_individuals = individuals[inx]
    return sorted_individuals


def fast_non_dominated_sort(values):
    """

    Parameters
    ----------
    values : np.array
    """
    num = len(values)
    s_2 = {}
    front = [np.array([], dtype=np.int32)]
    n = np.zeros(num, dtype=np.int32)
    num_multi_obj = values.shape[1]
    for i in range(num):
        # temp_ = np.delete(values, axis=0)
        temp_a = np.where(np.sum(values[i] <= values, axis=1) == num_multi_obj)[0]
        temp_b = np.where(np.sum(values[i] == values, axis=1) == num_multi_obj)[0]
        temp_c = np.where(np.sum(values[i] >= values, axis=1) == num_multi_obj)[0]
        temp_a = np.delete(temp_a, np.where(temp_a == i))
        temp_c = np.delete(temp_c, np.where(temp_c == i))
        if len(temp_b) != 0:
            for each in temp_b:
                temp_a = np.delete(temp_a, np.where(temp_a == each)[0])
                temp_b = np.delete(temp_b, np.where(temp_b == each)[0])
        the_dominated = temp_a
        the_dominator = temp_c
        n[i] += len(the_dominator)
        if len(the_dominator) == 0:
            front[0] = np.append(front[0], i)
            n[the_dominated] -= 1
        else:
            s_2[i] = the_dominated
    j = 1
    inx = np.where(n == 0)[0]
    set_used = set(front[0])
    front.append(np.array(list(set(inx) - set_used)))
    set_used = set_used.union(set(front[1]))
    while len(inx) < num:
        for each in front[j]:
            if len(s_2[each]) > 0:
                n[s_2[each]] -= 1
        inx = np.where(n == 0)[0]
        front.append(np.array(list(set(inx) - set_used)))
        j += 1
        set_used = set_used.union(set(front[j]))
    return front


def crowding_distance(values, front_i, max_min, all_sort):
    distance = np.array([])
    sorted_result = sort_by_values(front_i, values, all_sort)
    len_ = sorted_result.shape[1]
    for each in front_i:
        inx = np.where(sorted_result == each)[1]
        temp_dis = 0
        beginning_or_end = (inx == 0) + (inx == len_ - 1)
        if any(beginning_or_end):
            temp_dis += 500 * np.sum(beginning_or_end)
        column = np.where(beginning_or_end == 0)[0]
        mid = inx[column]
        temp_dis += np.sum((values[sorted_result[column, mid+1], column] -
                            values[sorted_result[column, mid-1], column]) / max_min[column])
        distance = np.append(distance, temp_dis)

    """old, delete"""
    # distance = [0 for i in range(len(front_i))]
    # for k in range(0, len(front_i)):
    #     j = front_i[k]
    #     index = [np.where(sorted_result == j)][0][1]
    #     if index[0] == 0 or index[0] == len(front_i) - 1:
    #         distance[k] += 500
    #     else:
    #         distance[k] += (values[sorted_result[0][index[0]+1], 0] - values[sorted_result[0][index[0]-1], 0])/(max_min[0])
    #     if index[1] == 0 or index[1] == len(front_i) - 1:
    #         distance[k] += 500
    #     else:
    #         distance[k] += (values[sorted_result[1][index[1]+1], 1] - values[sorted_result[1][index[1]-1], 1])/(max_min[1])
    #     if index[2] == 0 or index[2] == len(front_i) - 1:
    #         distance[k] += 500
    #     else:
    #         distance[k] += (values[sorted_result[2][index[2]+1], 2] - values[sorted_result[2][index[2]-1], 2])/(max_min[2])
    #     if index[3] == 0 or index[3] == len(front_i) - 1:
    #         distance[k] += 500
    #     else:
    #         distance[k] += (values[sorted_result[3][index[3]+1], 3] - values[sorted_result[3][index[3]-1], 3])/(max_min[3])
    #     if index[4] == 0 or index[4] == len(front_i) - 1:
    #         distance[k] += 500
    #     else:
    #         distance[k] += (values[sorted_result[4][index[4]+1], 4] - values[sorted_result[4][index[4]-1], 4])/(max_min[4])
    return distance


def sort_by_values(front_i, values, all_sort):

    needless = np.array(list(set(all_sort[:, 0]) - set(front_i)))
    s = all_sort.T.flatten()
    index = np.where(np.in1d(s, needless))[0]
    sorted_temp = np.delete(s, index)
    sorted_arr = sorted_temp.reshape(values.shape[1], -1)
    return sorted_arr


# def index_of(a, list_1):
#     for i in range(0, len(list_1)):
#         if list_1[i] == a:
#             return i
#     return -1


def alpha_level_sort(population, individuals, _so_param):
    b = np.array([0.2, 0.4, 20, 15, 5])
    if population.generation >= population.gen_max / 2:
        alpha = 1
    else:
        alpha = eval(_so_param)(population)
    population.sort_cache = alpha
    sort_basis = np.zeros(len(individuals))
    constraint_temp = []
    fitness_weight = []
    for individual in individuals:
        if individual.sort_basis is None:
            individual.sort_basis = min(1 - np.array(individual.constraint) / b)
            if individual.sort_basis <= 0:
                individual.sort_basis = 0.0001
        constraint_temp.append(individual.sort_basis)
        fitness_weight.append(individual.fitness_wight_factor)
    constraint_temp = np.array(constraint_temp)
    fitness_weight = np.array(fitness_weight)
    inx = np.where((constraint_temp < alpha) == 1)[0]
    sort_basis[inx] = 1000 / constraint_temp[inx]
    sort_basis += fitness_weight
    return individuals[np.argsort(sort_basis)]


def compute_alpha_1(population):
    return population.generation / population.gen_max


def compute_alpha_2(population):
    return (1 + np.exp(- population.generation +
                       population.gen_max / 2)) ** -1


def compute_alpha_3(population):
    return population.generation ** 2 / population.gen_max ** 2


def compute_alpha_4(population):
    return population.generation ** 3 / population.gen_max ** 3


def compute_alpha_5(population):
    return (1 - 0.01) * population.sort_cache + 0.01


def compute_alpha_6(population):
    return (1 - 0.05) * population.sort_cache + 0.05


def compute_alpha_7(population):
    return (1 + np.exp(4 - 10 * population.generation
                       / population.gen_max)) ** -1


def compute_alpha_8(population):
    return (1 + np.exp(4 - 20 * population.generation
                       / population.gen_max)) ** -1


"""old alpha , delete"""
# def compute_alpha(generation, gen_max, _so_param, alpha_pre):
#     if generation >= gen_max / 2:
#         return 1
#     if _so_param == '000':
#         alpha = generation / gen_max
#     elif _so_param == '001':
#         alpha = (1 + np.exp(- generation + gen_max/2)) ** -1
#     elif _so_param == '010':
#         alpha = generation ** 2 / gen_max ** 2
#     elif _so_param == '011':
#         alpha = generation ** 3 / gen_max ** 3
#     elif _so_param == '100':
#         alpha = (1 - 0.01) * alpha_pre + 0.01
#     elif _so_param == '101':
#         alpha = (1 - 0.05) * alpha_pre + 0.05
#     elif _so_param == '110':
#         alpha = (1 + np.exp(4 - 10 * generation / gen_max)) ** -1
#     else:
#         alpha = (1 + np.exp(4 - 20 * generation / gen_max)) ** -1
#     return alpha


def num_of_un_constraint(population, individuals, _so_param):
    sort_basis = np.array([])
    for individual in individuals:
        if individual.sort_basis is None:
            individual.sort_basis = un_constraint(individual.fitness_wight_factor, individual.constraint)
        sort_basis = np.append(sort_basis, individual.sort_basis)
    return individuals[np.argsort(sort_basis)]


def un_constraint(fit, constraint):
    count = 0
    for con in constraint:
        if con != 0:
            count += 1
    return count * 10000 + fit


"""old_select, delete"""
# def selection(population, _se, _se_param):
#     if _se == '00':
#         if population.selection_param is None:
#             if _se_param == '00':
#                 population.selection_param = ["truncation", 0.3]
#             if _se_param == '01':
#                 population.selection_param = ["truncation", 0.5]
#             if _se_param == '10':
#                 population.selection_param = ["truncation", 0.7]
#             if _se_param == '11':
#                 population.selection_param = ["truncation", 0.9]
#         individual = truncation_selection(population)
#     elif _se == '01':
#         if population.selection_param is None:
#             if _se_param == '00':
#                 population.selection_param = ["tournament", 1]
#             if _se_param == '01':
#                 population.selection_param = ["tournament", 2]
#             if _se_param == '10':
#                 population.selection_param = ["tournament", 3]
#             if _se_param == '11':
#                 population.selection_param = ["tournament", 4]
#         individual = tournament_selection(population)
#     elif _se == '10':
#         if population.selection_param is None:
#             if _se_param == '00':
#                 population.selection_param = ['fitness_no_duplication']
#                 population.selection_cache = []
#             if _se_param == '01':
#                 population.selection_param = ['fitness']
#             if _se_param == '10':
#                 population.selection_param = ['ranking_system_no_duplication']
#                 population.selection_cache = []
#             if _se_param == '11':
#                 population.selection_param = ['ranking_system']
#         individual = roulette_wheel_selection(population)
#     else:
#         if population.selection_param is None:
#             population.selection_cache = []
#             if _se_param == '00':
#                 population.selection_param = ['sus', 2]
#             if _se_param == '01':
#                 population.selection_param = ['sus', 3]
#             if _se_param == '10':
#                 population.selection_param = ['sus', 4]
#             if _se_param == '11':
#                 population.selection_param = ['sus', 5]
#         individual = stochastic_universal_selection(population)
#     return individual


def truncation_selection(population, select_pool, se_param):
    individual = select_pool[population.selection_cache]
    if population.selection_cache < se_param * select_pool.size:
        population.selection_cache += 1
    else:
        population.selection_cache = 0
    return individual


def tournament_selection(population, select_pool, se_param):
    cache = []
    for i in range(se_param):
        cache.append(np.random.randint(0, select_pool.size))
    return select_pool[min(cache)]


def fitness_prob(select_pool):
    total = sum([1 / select_pool[x].fitness_wight_factor for x in range(select_pool.size)])
    prob = np.array([(1 / select_pool[0].fitness_wight_factor) / total])
    for x in range(1, select_pool.size):
        prob = np.append(prob, (1 / select_pool[x].fitness_wight_factor) / total + prob[x - 1])
    return prob


def fitness_no_duplication(population, select_pool):
    return fitness_prob(select_pool), 0


def fitness_allow_duplication(select_pool):
    return fitness_prob(select_pool), 1


def ranking_system_prob(population, select_pool):
    if select_pool.size == population.num_individual:
        if population.rank_probability is None:
            rank_score = eval(population.soft_inf.rank)(population.num_individual)
            total = np.sum(rank_score)
            prob = rank_score / total
            for x in range(1, select_pool.size):
                prob[x] += prob[x - 1]
            population.rank_probability = prob
        return population.rank_probability
    else:
        rank_score = eval(population.soft_inf.rank)(select_pool.size)
        total = np.sum(rank_score)
        prob = rank_score / total
        for x in range(1, select_pool.size):
            prob[x] += prob[x - 1]
    return prob


def ranking_system_no_duplication(population, select_pool):
    return ranking_system_prob(population, select_pool), 0


def ranking_system_allow_duplication(population, select_pool):
    return ranking_system_prob(population, select_pool), 1


def roulette_wheel_selection(population, select_pool, se_param):
    # if population.selection_param[0] == 'fitness_no_duplication' or population.selection_param[0] == 'fitness':
    #     if len(population.selection_param) == 1:
    #         population.selection_param.append(sum([1 / population.data[x].fitness_wight_factor for x in range(population.num_individual)]))
    #         population.selection_param.append([(1 / population.data[x].fitness_wight_factor) /
    #                                            population.selection_param[1] for x in range(population.num_individual)])
    # else:
    #     if len(population.selection_param) == 1:
    #         population.selection_param.append(sum(population.rank_score))
    #         population.selection_param.append(population.rank_score / population.selection_param[1])
    if population.selection_cache == 0:
        population.selection_cache = eval(se_param)(population, select_pool)
    return select_pool[rws_selection_result(population.selection_cache[0])]
    # while True:
    #     index = rws_selection_result(population.num_individual, population.selection_param[2])
    #     if population.selection_param[0] == 'fitness' or population.selection_param[0] == 'ranking_system':
    #         return population.data[index]
    #     if len(population.selection_cache) >= population.num_individual * 0.7:
    #         population.selection_cache = []
    #     if index not in population.selection_cache:
    #         population.selection_cache.append(index)
    #         return population.data[index]


"old_rws, delete"
# def rws_selection_result(selection_basis):
#     index = 0
#     rand_0_1 = np.random.rand()
#     for i in range(selection_basis.size):
#         rand_0_1 -= selection_basis[i]
#         if rand_0_1 <= 0:
#             index = i
#             break
#     return index


def rws_selection_result(selection_basis):
    rand_0_1 = np.random.rand()
    return np.argmax(selection_basis >= rand_0_1)


def stochastic_universal_selection(population, select_pool, se_param):
    if population.selection_cache == 0:
        population.selection_cache = [fitness_prob(select_pool), []]
    if not population.selection_cache[1]:
        rand_origin = np.random.rand()/se_param
        rand_ultimate = np.arange(rand_origin, 1, 1 / se_param)
        for each in rand_ultimate:
            population.selection_cache[1].append(np.argmax(population.selection_cache[0] >= each))
    return select_pool[population.selection_cache[1].pop()]

    # if len(population.selection_param) == 2:
    #     population.selection_param.append(population.rank_score / sum(population.rank_score))
    # if not population.selection_cache:
    #     count = population.selection_param[1]
    #     rand_0_1 = np.random.rand() / count
    #     for i in range(population.num_individual):
    #         rand_0_1 -= population.selection_param[2][i]
    #         if rand_0_1 <= 0:
    #             population.selection_cache.append(population.data[i])
    #             count -= 1
    #             if count != 0:
    #                 rand_0_1 += 1 / population.selection_param[1]
    #             else:
    #                 break
    # individual = population.selection_cache.pop(0)
    # return individual


if __name__ == "__main__":
    # global_map = test_map(10, 10, 10)
    # a = time.clock()
    # p = test_population(global_map)
    # sorted_individual = sort(p, p.data, '00', '000')
    # print(time.clock() - a)
    # a = selection(p, '10', '10')
    a = time.clock()
    # for i in range(100):
    #     rws_selection_result(selection_basis)
    # print(time.clock() - a)