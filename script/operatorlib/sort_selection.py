from initialize import *
import math


def sort(population, _so, _so_param):
    if _so == '00':
        penalty_sort(population, _so_param)
    elif _so == '01':
        non_dominated_sort(population, _so_param)
    elif _so == '10':
        alpha_level_sort(population, _so_param)
    else:
        num_of_un_constraint(population, _so_param)
    population.data.sort(key=lambda x: x.sort_basis)
    for i in range(population.num_individual):
        population.data[i].sort_basis = None


def penalty_sort(population, _so_param):
    for individual in population.data:
        if individual.sort_basis is None:
            individual.sort_basis = penalty_function(population.gen_max, individual.fitness_wight_factor, individual.constraint, population.generation, _so_param)


def penalty_function(gen_max, fit, constraint, generation, _so_param):
    penalty = 0
    for con in constraint:
        penalty += con ** 2
    if _so_param == '000':
        r_k = 1000 / gen_max * generation
    elif _so_param == '001':
        r_k = 5000 / gen_max * generation
    elif _so_param == '010':
        r_k = 1000 * (1 + np.exp(- generation + gen_max/2)) ** -1 #sigmoid function
    elif _so_param == '011':
        r_k = 5000 * (1 + np.exp(- generation + gen_max/2)) ** -1
    elif _so_param == '100':
        r_k = 1000 / (gen_max ** 2) * generation ** 2 #ex
    elif _so_param == '101':
        r_k = 5000 / (gen_max ** 2) * generation ** 2
    elif _so_param == '110':
        r_k = 1000 / (gen_max ** 3) * generation ** 3
    else:
        r_k = 5000 / (gen_max ** 3) * generation ** 3
    return fit + penalty * r_k


def non_dominated_sort(population, _so_param):
    values_1 = [population.data[f].fitness[0] for f in range(len(population.data))]
    values_2 = [population.data[f].fitness[1] for f in range(len(population.data))]
    values_3 = [population.data[f].fitness[2] for f in range(len(population.data))]
    values_4 = [population.data[f].fitness[3] for f in range(len(population.data))]
    values_5 = [population.data[f].fitness[4] for f in range(len(population.data))]
    front = fast_non_dominated_sort(values_1, values_2, values_3, values_4, values_5)
    crowding_distance_values = []
    max_min_obj = [max(values_1) - min(values_1), max(values_2) - min(values_2), max(values_3) - min(values_3),
                   max(values_4) - min(values_4), max(values_5) - min(values_5)]
    for i in range(len(max_min_obj)):
        if max_min_obj[i] == 0:
            max_min_obj[i] = 0.01
    for i in range(0, len(front)):
        crowding_distance_values.append(crowding_distance(values_1, values_2, values_3, values_4, values_5, front[i], max_min_obj))
    count = 0
    for i in range(0,len(front)):
        front_2 = [index_of(front[i][j], front[i]) for j in range(0,len(front[i]))]
        front_22 = sort_by_values(front_2[:], crowding_distance_values[i][:])
        front_out = [front[i][front_22[j]] for j in range(0,len(front[i]))]
        front_out.reverse()
        for index in front_out:
            population.data[index].sort_basis = count
            count += 0.01
        count = int(count + 1)


def fast_non_dominated_sort(values_1, values_2, values_3, values_4, values_5):
    s = [[] for i in range(0, len(values_1))]
    front = [[]]
    n=[0 for i in range(0,len(values_1))]
    rank = [0 for i in range(0, len(values_1))]
    for i in range(0, len(values_1)):
        s[i] = []
        n[i] = 0
        for q in range(0, len(values_1)):
            if values_1[i] < values_1[q] and values_2[i] <= values_2[q] \
                    and values_3[i] <= values_3[q] and values_4[i] <= values_4[q] and values_5[i] <= values_5[q]:
                if q not in s[i]:
                    s[i].append(q)
            elif values_1[i] > values_1[q] and values_2[i] >= values_2[q] and values_3[i] >= \
                    values_3[q] and values_4[i] >= values_4[q] and values_5[i] >= values_5[q]:
                n[i] = n[i] + 1
        if n[i] == 0:
            rank[i] = 0
            if i not in front[0]:
                front[0].append(i)
    j = 0
    while front[j]:
        Q=[]
        for i in front[j]:
            k: int
            for k in s[i]:
                n[k] =n[k] - 1
                if n[k] == 0:
                    rank[k]=j+1
                    if k not in Q:
                        Q.append(k)
        j = j+1
        front.append(Q)
    del front[len(front)-1]
    return front


def crowding_distance(values_1, values_2, values_3, values_4, values_5, list_1, max_min):
    distance = [0 for i in range(0,len(list_1))]
    sorted1 = sort_by_values(list_1, values_1[:])
    sorted2 = sort_by_values(list_1, values_2[:])
    sorted3 = sort_by_values(list_1, values_3[:])
    sorted4 = sort_by_values(list_1, values_4[:])
    sorted5 = sort_by_values(list_1, values_5[:])
    for k in range(0, len(list_1)):
        j = list_1[k]
        index = [sorted1.index(j), sorted2.index(j), sorted3.index(j), sorted4.index(j), sorted5.index(j)]
        if index[0] == 0 or index[0] == len(list_1) -1:
            distance[k] += 500
        else:
            distance[k] += (values_1[sorted1[index[0]+1]] - values_1[sorted1[index[0]-1]])/(max_min[0])
        if index[1] == 0 or index[1] == len(list_1) - 1:
            distance[k] += 500
        else:
            distance[k] += (values_2[sorted2[index[1]+1]] - values_2[sorted2[index[1]-1]])/(max_min[1])
        if index[2] == 0 or index[2] == len(list_1) - 1:
            distance[k] += 500
        else:
            distance[k] += (values_3[sorted3[index[2]+1]] - values_3[sorted3[index[2]-1]])/(max_min[2])
        if index[3] == 0 or index[3] == len(list_1) - 1:
            distance[k] += 500
        else:
            distance[k] += (values_4[sorted4[index[3]+1]] - values_4[sorted4[index[3]-1]])/(max_min[3])
        if index[4] == 0 or index[4] == len(list_1) - 1:
            distance[k] += 500
        else:
            distance[k] += (values_5[sorted5[index[4]+1]] - values_5[sorted5[index[4]-1]])/(max_min[4])
    return distance


def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list)!= len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def index_of(a, list_1):
    for i in range(0, len(list_1)):
        if list_1[i] == a:
            return i
    return -1


def alpha_level_sort(population, _so_param):
    b = np.array([0.2, 0.4, 20, 15, 5])
    alpha = compute_alpha(population.generation, population.gen_max, _so_param, population.alpha)
    population.alpha = alpha
    for individual in population.data:
        if individual.constraint_satisfaction_level is None:
            individual.constraint_satisfaction_level = min(1 - np.array(individual.constraint)/b)
            if individual.constraint_satisfaction_level <= 0:
                individual.constraint_satisfaction_level = 0.0001
        if individual.constraint_satisfaction_level >= alpha:
            individual.sort_basis = individual.fitness_wight_factor
        else:
            individual.sort_basis = individual.fitness_wight_factor + 1000 / individual.constraint_satisfaction_level


def compute_alpha(generation, gen_max, _so_param, alpha_pre):
    if generation >= gen_max / 2:
        return 1
    if _so_param == '000':
        alpha = generation / gen_max
    elif _so_param == '001':
        alpha = (1 + np.exp(- generation + gen_max/2)) ** -1
    elif _so_param == '010':
        alpha =  generation ** 2 / gen_max ** 2
    elif _so_param == '011':
        alpha = generation ** 3 / gen_max ** 3
    elif _so_param == '100':
        alpha = (1 - 0.01) * alpha_pre + 0.01
    elif _so_param == '101':
        alpha = (1 - 0.05) * alpha_pre + 0.05
    elif _so_param == '110':
        alpha = (1 + np.exp(4 - 10 * generation / gen_max)) ** -1
    else:
        alpha = (1 + np.exp(4 - 20 * generation / gen_max)) ** -1
    return alpha


def num_of_un_constraint(population, _so_param):
    for individual in population.data:
        if individual.sort_basis is None:
            individual.sort_basis = un_constraint(individual.fitness_wight_factor, individual.constraint)


def un_constraint(fit, constraint):
    count = 0
    for con in constraint:
        if con != 0:
            count += 1
    return count * 10000 + fit


def selection(population, _se, _se_param):
    if _se == '00':
        if population.selection_param is None:
            if _se_param == '00':
                population.selection_param = ["truncation", 0.3]
            if _se_param == '01':
                population.selection_param = ["truncation", 0.5]
            if _se_param == '10':
                population.selection_param = ["truncation", 0.7]
            if _se_param == '11':
                population.selection_param = ["truncation", 0.9]
        individual = truncation_selection(population)
    elif _se == '01':
        if population.selection_param is None:
            if _se_param == '00':
                population.selection_param = ["tournament", 1]
            if _se_param == '01':
                population.selection_param = ["tournament", 2]
            if _se_param == '10':
                population.selection_param = ["tournament", 3]
            if _se_param == '11':
                population.selection_param = ["tournament", 4]
        individual = tournament_selection(population)
    elif _se == '10':
        if population.selection_param is None:
            if _se_param == '00':
                population.selection_param = ['fitness_no_duplication']
                population.selection_cache = []
            if _se_param == '01':
                population.selection_param = ['fitness']
            if _se_param == '10':
                population.selection_param = ['ranking_system_no_duplication']
                population.selection_cache = []
            if _se_param == '11':
                population.selection_param = ['ranking_system']
        individual = roulette_wheel_selection(population)
    else:
        if population.selection_param is None:
            population.selection_cache = []
            if _se_param == '00':
                population.selection_param = ['sus', 2]
            if _se_param == '01':
                population.selection_param = ['sus', 3]
            if _se_param == '10':
                population.selection_param = ['sus', 4]
            if _se_param == '11':
                population.selection_param = ['sus', 5]
        individual = stochastic_universal_selection(population)
    return individual


def truncation_selection(population):
    individual = population.data[population.selection_cache]
    if population.selection_cache < population.selection_param[1] * population.num_individual:
        population.selection_cache += 1
    else:
        population.selection_cache = 0
    return individual


def tournament_selection(population):
    cache = []
    for i in range(population.selection_param[1]):
        cache.append(np.random.randint(0, population.num_individual))
    return population.data[min(cache)]


def roulette_wheel_selection(population):
    if population.selection_param[0] == 'fitness_no_duplication' or population.selection_param[0] == 'fitness':
        if len(population.selection_param) == 1:
            population.selection_param.append(sum([1 / population.data[x].fitness_wight_factor for x in range(population.num_individual)]))
            population.selection_param.append([(1 / population.data[x].fitness_wight_factor) /
                                               population.selection_param[1] for x in range(population.num_individual)])
    else:
        if len(population.selection_param) == 1:
            population.selection_param.append(sum(population.rank_score))
            population.selection_param.append(population.rank_score / population.selection_param[1])

    while True:
        index = rws_selection_result(population.num_individual, population.selection_param[2])
        if population.selection_param[0] == 'fitness' or population.selection_param[0] == 'ranking_system':
            return population.data[index]
        if len(population.selection_cache) >= population.num_individual * 0.7:
            population.selection_cache = []
        if index not in population.selection_cache:
            population.selection_cache.append(index)
            return population.data[index]


def rws_selection_result(num, selection_basis):
    index = 0
    rand_0_1 = np.random.rand()
    for i in range(num):
        rand_0_1 -= selection_basis[i]
        if rand_0_1 <= 0:
            index = i
            break
    return index


def stochastic_universal_selection(population):
    if len(population.selection_param) == 2:
        population.selection_param.append(population.rank_score / sum(population.rank_score))
    if not population.selection_cache:
        count = population.selection_param[1]
        rand_0_1 = np.random.rand() / count
        for i in range(population.num_individual):
            rand_0_1 -= population.selection_param[2][i]
            if rand_0_1 <= 0:
                population.selection_cache.append(population.data[i])
                count -= 1
                if count != 0:
                    rand_0_1 += 1 / population.selection_param[1]
                else:
                    break
    individual = population.selection_cache.pop(0)
    return individual


if __name__ == "__main__":
    p = test()
    sort(p, '00', '000')
    a = selection(p, '10', '10')