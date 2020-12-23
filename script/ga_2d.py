from math import sqrt
import random
from typing import Any, Union

from scipy.special import comb
import matplotlib.pyplot as plt
import numpy as np
import time
import rospy
from generatemap import terrain

class individual:
    def __init__(self, points, start, goal):
        """
        Create a member of the population
        nPoints: the number of values per individual
        start: start point
        goal: goal point
        """
        self.points = points
        self.path = path
        self.fitness = fitness(self.trajectory, start, goal, obstacles)


def bezier_curve(points, nTimes=80):
    """
   Given a set of control points, return the
   bezier curve defined by the control points.
    """
    nPoints = len(points)
    # Creating random points for control points
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    xvals.astype(int)
    yvals.astype(int)

    # curve = (map(lambda x, y:(x, y), xvals, yvals))
    path = [[xvals[i], yvals[i]] for i in range(0, len(xvals))]
    path.reverse()
    '''
    plt.plot(xvals, yvals)
    plt.plot(xPoints, yPoints, "ro")
    for nr in range(len(xPoints)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.show()
    '''

    return path


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def Population(population_individuals, nPoints, start, goal, map_size, obstacles):
    """
    Create a number of individuals (i.e. a population).
    """
    population_pre = []
    for i in range(population_individuals * 2):
        points = []
        points.append(start)
        for j in range(1, nPoints-1):
            points.append(np.random.rand(2) * mapsize)
        points.append(goal)
        population_pre.append(individual(points, start, goal, obstacles))

    # print (population_pre)

    graded = merge_sort(population_pre)

    return graded[0:len(graded)//2]


def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    middle = len(lists)//2
    left = merge_sort(lists[:middle])
    right = merge_sort(lists[middle:])
    return merge(left, right)


def merge(left, right):
    c = []
    h = j = 0
    while j < len(left) and h < len(right):
        if left[j].fitness <= right[h].fitness:
            c.append(left[j])
            j += 1
        else:
            c.append(right[h])
            h += 1
    while j < len(left):
        c.append(left[j])
        j += 1
    while h < len(right):
        c.append(right[h])
        h += 1
    return c


def fitness(trajectory, start, goal, obstacles):
    """
    Determine the fitness of an individual. Lower is better.

    individual: the individual to evaluate
    start: start point
    goal: goal point
    vertexs_list: a list of vertexs of obstacles
    """
    # TODO
    # Fix this fitness function with the energy function for path planning
    previous_point = (start[0], start[1])
    min_curvature = 5
    dist_optimal = sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

    # print (individual)
    # TODO
    # more general function for all possible constrain
    a, b = np.split(obstacles, 2, axis=1)
    inside_point_num = inside_circle([i[0] for i in trajectory], [i[1] for i in trajectory], a, b)
    fitness_vertex = inside_point_num * 1000

    # Fitness about euclid distance
    fitness_euclid = 0
    for point in trajectory:
        fitness_euclid += sqrt((point[0] - previous_point[0]) ** 2 + (point[1] - previous_point[1]) ** 2)
        previous_point = point
        # print (fitness_euclid)

    curve = trajectory

    x = np.array(curve)[:, 0]
    y = np.array(curve)[:, 1]

    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

    # print (curvature)
    fitness_curvature = 0
    for curve_value in np.ndindex(curvature.shape):
        # enum
        if curvature[curve_value] > min_curvature:
            # print('the curvature is higher than min_curvature')
            fitness_curvature += 1000

    fitness_final = sqrt(
        ((fitness_curvature / min_curvature) ** 2) + (fitness_euclid / dist_optimal) ** 2 + fitness_vertex ** 2)

    # print ('final fitness is equal to: ', fitness_final)
    return fitness_final


def evolve(parents, start, goal, obstacles, retain=1, random_select=0.05, mutate=0.35):
    """
    evolution operations, include selection, mutation, crossover

    pop: one generation
    vertexs_list: a list of vertexs of obstacles
    retain: percentage of individuals kept every generation
    random_select: select rate for increase diversity
    mutate: mutation rate
    crossover_pos: crossover's position
    """

    # TODO
    ''' 
    # randomly add other individuals to promote genetic diversity
    for individual in graded[:retain_length]:
        if random_select < random.random():
            parents.append(individual)
    '''

    # crossover parents to create children
    desired_length = int(len(parents) * retain)
    children = []
    select_list = np.linspace(0.0, 1.0, len(parents))
    select_sum = sum(select_list)
    select_probability = list(select_list / select_sum)
    select_probability.reverse()
    roulette_rate = np.cumsum(np.array(select_probability), dtype=float)
    while len(children) < desired_length:
        # TODO selection
        male = binary_search(random.random(), roulette_rate)
        female = binary_search(random.random(), roulette_rate)

        if male != female:
            male = parents[male].points
            female = parents[female].points
            crossover_pos = random.randint(0, len(male) - 1)
            child = male[:crossover_pos] + female[crossover_pos:]
            children.append(individual(child, start, goal, obstacles))

    # mutate some individuals
    for each in children:
        if mutate > random.random():
            pos_to_mutate = random.randint(1, len(each.points) - 2)
            # TODO
            # this mutation is not ideal, because it can hardly find a property solution
            each.points[pos_to_mutate] = (random.uniform(start[0], goal[0]), random.uniform(start[1], goal[1]))
    parents.extend(children)
    parents = merge_sort(parents)
    a = len(parents) // 2

    return parents[:a]


def binary_search(number, array):
    length = len(array)
    if length == 1:
        return 0
    length = int(length/2)
    if array[length - 1] <= number <= array[length]:
        return length
    if array[length-1] <= number and array[length] <= number:
        return binary_search(number, array[length:]) + length
    else:
        return binary_search(number, array[0:length])


def ga_execute(start, goal, population, generation, nPoints, map_size, obstacles):
    """
    entrance of genetic algorithm
    this function is executed in supervisor

    start: start point
    goal: goal point
    """
    pop = Population(population, nPoints, start, goal, map_size, obstacles)
    score = []
    g = 0
    total = 0
    while g < generation:
        t = time.time()
        pop = evolve(pop, start, goal, obstacles)
        score.append(pop[0].fitness)
        g += 1
        t_ = time.time() - t
        print("generation", g, ", the best fitness is ", pop[0].fitness, "time cost:", t_)
        total += t_
    print("total cost: ", total)
    # select the best individual
    best_individual = pop[0]
    '''
    xs = [x[0] for x in best_individual]
    ys = [x[1] for x in best_individual]
    plt.plot(xs, ys)
    '''

    return score, best_individual


def inside_circle(x, y, a, b, r=1.05):
    n = 0
    for i in range(len(x)):
        for j in range(len(a)):
            if (x[i] - a[j]) * (x[i] - a[j]) + (y[i] - b[j]) * (y[i] - b[j]) < r * r:
                n += 1
    return n


def getCurve(map_size, obstacles, nPoints):
    # points = []
    # for p in range(nPoints + 1):
    #     points.append((np.random.random() * np.random.uniform(1, map_size),
    #                    np.random.random() * np.random.uniform(1, map_size)))

    start = np.array([1, 1])
    goal = np.array([mapsize-1, map_size-1])
    population = 20
    generation = 5
    best_fitness, best_individual = ga_execute(start, goal, population, generation, nPoints, map_size, obstacles)

    xs = [x[0] for x in best_individual.trajectory]
    ys = [x[1] for x in best_individual.trajectory]


    # plot trajectory
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker='o', markersize=2.5)
    ax.axis('equal')
    circle = []
    for i in range(len(obstacles)):
        circle = plt.Circle(obstacles[i], 1, color='black')
        ax.add_artist(circle)

    ax.set_xbound(lower=0, upper=map_size + 1)
    ax.set_ybound(lower=0, upper=map_size + 1)

    plt.suptitle('Genetic Algorith_Path Planning - m', fontsize=14, color='black')
    plt.xlabel('X Position', fontsize=12, color='black')
    plt.ylabel('Y Position', fontsize=12, color='black')

    plt.grid(True)

    plt.show()

    # plot fitness
    plt.plot(best_fitness)
    plt.suptitle("the change of fitness over generation", fontsize=14, color='black')
    plt.xlabel("generation", fontsize=12, color='black')
    plt.ylabel("fitness", fontsize=12, color='black')
    plt.grid(True)

    plt.show()
    return best_individual, best_fitness


if __name__ == '__main__':
    # mapsize = 16
    # # obs = np.array([[ 5.41022686,  8.61243773],
    # #    [10.48489417,  5.60397119],
    # #    [ 2.86068703,  5.14619389],
    # #    [ 3.22309725,  8.56178768],
    # #    [12.62500902, 13.14239391],
    # #    [13.95180637,  6.3945937 ],
    # #    [ 7.13261747,  2.86569717],
    # #    [ 3.94899431,  1.17146981],
    # #    [14.03970873,  9.43937267],
    # #    [ 2.76140657, 12.00313488],
    # #    [12.14357054,  9.62369838],
    # #    [ 2.21195517,  4.34068619],
    # #    [ 4.07020724,  6.75652512],
    # #    [10.15570869, 10.03985702],
    # #    [ 2.3573808 ,  1.6167306 ],
    # #    [ 7.12685735,  8.58548299],
    # #    [ 7.02469972,  2.59285701],
    # #    [ 9.48494011,  7.29211685],
    # #    [ 4.45415489, 10.46243259],
    # #    [ 2.54677276,  4.29169719]])
    # obs = np.random.rand(10, 2) * mapsize
    # n_points = 5
    # solution, fitness = getCurve(mapsize, obs, n_points)
    terrain()

