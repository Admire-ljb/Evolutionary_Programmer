from math import sqrt
import random
from typing import Any, Union
from calculate import *
import matplotlib.pyplot as plt
import numpy as np
import time
import rospy


class Individual:
    def __init__(self, pop):
        """
        Create an individual
        pop: the population, recording some shared imformation
        map: the global map consisting of terrain, radar, missile and no-fly zone
        start: start point
        goal: goal point
        curve_type: the curve using to smooth the trajectory
        num_cp: the number of control_points
        coordinates: the rotate coordinates of ST
        """
        self.control_points = initialize(pop.global_map, pop.num_cp, pop.rotation_matrix,
                                              pop.start, pop.goal_r)
        self.trajectory = self.curve(pop.curve_type)
        # self.fitness = fitness

    def curve(self, cv):
        """"""
        if cv == '00':
            trajectory = six_bezier(self.control_points)
        elif cv == '01':
            trajectory = b_spline(self.control_points)
        elif cv == '10':
            trajectory = rts(self.control_points)
        else:
            trajectory = tangent_circle_curve(self.control_points)
        return trajectory


class Population:
    """
    Create a number of individuals (i.e. a population).
    """
    def __init__(self, num_individual, start, goal, global_map, curve_type, cp):
        self.data = list()
        self.start = start
        self.goal = goal
        self.curve_type = curve_type
        self.rotation_matrix = self.rotate_coordinate()
        self.goal_r = rotation2st(start, goal, self.rotation_matrix)
        self.cp = cp
        self.global_map = global_map
        self.num_cp = self.decode_cp()
        self.upper_bound, self.lower_bound = self.y_boundary(global_map)
        for i in range(num_individual):
            self.data.append(Individual(self))

    def y_boundary(self, global_map):
        delta_d = 1
        y_upper = []
        y_lower = []
        for each in global_map.radar:
            o = rotation2st(self.start, each.center, self.rotation_matrix)
            if -each.radius < o[0] < self.goal_r[0] + each.radius:
                y_upper.append(o[1]+each.radius)
                y_lower.append(o[1]-each.radius)

        for each in global_map.missile:
            o = rotation2st(self.start, each.center, self.rotation_matrix)
            if -each.radius < o[0] < self.goal_r[0] + each.radius:
                y_upper.append(o[1] + each.radius)
                y_lower.append(o[1] - each.radius)
        points = []
        for each in global_map.nfz:
            points.append(rotation2st(self.start, np.array([each.x_min, each.y_min, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_min, each.y_max, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_max, each.y_min, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_max, each.y_max, 0]), self.rotation_matrix))
        for i in points:
            if 0 < i[0] < self.goal_r[0]:
                y_upper.append(i[1])
                y_lower.append(i[1])
        return max(max(y_upper), 0) + delta_d, min(min(y_lower), 0) - delta_d

    def rotate_coordinate(self):
        theta = np.arctan((self.goal[1] - self.start[1])/(self.goal[0]-self.start[0]))
        if self.goal[0] < self.start[0]:
            theta += np.pi
        if self.goal[0] == self.start[0]:
            theta += np.pi * bool(self.start[1] > self.goal[1])
        rotation_matrix = np.array([[np.cos(theta),  np.sin(theta), 0],
                                    [- np.sin(theta), np.cos(theta), 0],
                                    [0,                0,            1]])
        return rotation_matrix

    def decode_cp(self):
        distance = np.linalg.norm(self.goal[0:2] - self.start[0:2])
        if self.cp == '00':
            len_points = distance // 10 + 2
        elif self.cp == '01':
            len_points = distance // 5 + 2
        elif self.cp == '10':
            len_points = distance // 2 + 2
        else:
            len_points = distance // 1 + 2
        return len_points



def six_bezier(control_points):
    trajectory = []
    for i in control_points:
        trajectory = []
        #TODO
    return trajectory

def b_spline(self):
    trajectory = []
    for i in self.control_points:
        trajectory = []
        #TODO
    return trajectory

def rts(self):
    trajectory = []
    for i in self.control_points:
        trajectory = []
        #TODO
    return trajectory

def tangent_circle_curve(self):
    trajectory = []
    for i in self.control_points:
        trajectory = []
        #TODO
    return trajectory


def initialize(global_map, num_cp, rotation_matrix, start, goal_r):
    """
    initialize the control_points
    """
    safe_height = 5
    points = list()
    start_r = [0, 0, start[2]]
    points.append(start_r)
    delta_l = goal_r[0] / num_cp
    for i in range(1, num_cp-1):
        temp = list()
        temp.append(delta_l*(i+1))
        if i == 1:
            temp.append(random.random() * delta_l*2 - delta_l)
        else:
            temp.append(random.random() * delta_l*6 - delta_l*3 + points[i-1][1])
        if i == 1:
            temp.append(points[0][2] + safe_height)
        else:
            temp.append(np.random.normal(global_map.terrain.map(int(temp[0]), int(temp[1])) -
                                         global_map.terrain.map(int(points[i-1][0]), int(points[i-1][1]))
                                         + points[i-1][2], delta_l/3))
        points.append(temp)
    points.append(list(goal_r))
    control_points = rotation2origin(start, np.array(points), rotation_matrix)
    return control_points


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
    mapsize = 16
    # obs = np.array([[ 5.41022686,  8.61243773],
    #    [10.48489417,  5.60397119],
    #    [ 2.86068703,  5.14619389],
    #    [ 3.22309725,  8.56178768],
    #    [12.62500902, 13.14239391],
    #    [13.95180637,  6.3945937 ],
    #    [ 7.13261747,  2.86569717],
    #    [ 3.94899431,  1.17146981],
    #    [14.03970873,  9.43937267],
    #    [ 2.76140657, 12.00313488],
    #    [12.14357054,  9.62369838],
    #    [ 2.21195517,  4.34068619],
    #    [ 4.07020724,  6.75652512],
    #    [10.15570869, 10.03985702],
    #    [ 2.3573808 ,  1.6167306 ],
    #    [ 7.12685735,  8.58548299],
    #    [ 7.02469972,  2.59285701],
    #    [ 9.48494011,  7.29211685],
    #    [ 4.45415489, 10.46243259],
    #    [ 2.54677276,  4.29169719]])
    obs = np.random.rand(10, 2) * mapsize
    n_points = 5
    solution, fitness = getCurve(mapsize, obs, n_points)


