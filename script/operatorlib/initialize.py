from math import sqrt
import random
from typing import Any, Union
from calculate import *
import matplotlib.pyplot as plt
import numpy as np
import time
import rospy
from main import *


class Individual:
    def __init__(self, pop, control_points=None):
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
        if control_points is None:
            self.control_points_r = np.array(initialize(pop.global_map, pop.num_cp,
                                             pop.start, pop.goal_r))
        else:
            self.control_points_r = control_points
        # self.control_points = rotation2origin(pop.start, self.control_points_r, pop.rotation_matrix)
        self.trajectory_r = self.curve(pop.curve_type, pop.len_individual)
        self.trajectory = rotation2origin(pop.start, self.trajectory_r, pop.rotation_matrix)
        self.fitness_wight_factor, self.fitness, self.constraint, self.sort_basis = \
            fitness(self.trajectory, self.trajectory_r, pop)
        self.constraint_satisfaction_level = None

    def curve(self, cv, n):
        """"""
        if cv == '00':
            trajectory_r = six_bezier(self.control_points_r, n)
        elif cv == '01':
            trajectory_r = b_spline(self.control_points_r, n)
        elif cv == '10':
            trajectory_r = rts(self.control_points_r, n)
        else:
            trajectory_r = tangent_circle_curve(self.control_points_r, n)
        return trajectory_r


class Population:
    """
    Create a number of individuals (i.e. a population).
    """
    def __init__(self, start, goal, g_map, _num_individual, curve_type, _cp, _rank_param, _elitism):
        self.data = list()
        self.num_individual = decode_n_i(_num_individual)
        self.start = start
        self.goal = goal
        self.curve_type = curve_type
        self.rotation_matrix = self.rotate_coordinate()
        self.goal_r = rotation2st(start, goal, self.rotation_matrix)
        self.global_map = g_map
        self.num_cp = self.decode_cp(_cp)
        self.upper_bound, self.lower_bound = self.y_boundary(g_map)
        self.len_individual = self.num_cp * 5
        self.gen_max = 10 # TODO
        self.generation = 0
        self.alpha = 0
        self.selection_param = None
        self.selection_cache = 0
        self.exploit_param = None
        self.explore_param = None
        self.elitism = decode_elitism(_elitism)
        for i in range(self.num_individual):
            self.data.append(Individual(self))
        self.rank_score = self.rank_system(_rank_param)

    def rank_system(self, _rank_param):
        rank_score = []
        if _rank_param == '00':
            rank_score = [1 / self.data[i].fitness_wight_factor for i in range(self.num_individual)]
        if _rank_param == '01':
            rank_score = [self.num_individual - i for i in range(self.num_individual)]
        if _rank_param == '10':
            rank_score = [np.sqrt(self.num_individual-i) for i in range(self.num_individual)]
        if _rank_param == '11':
            rank_score = [(self.num_individual-i) ** 2 for i in range(self.num_individual)]
        return np.array(rank_score)

    def y_boundary(self, g_map):
        delta_d = 1
        y_upper = []
        y_lower = []
        for each in g_map.radar:
            o = rotation2st(self.start, each.center, self.rotation_matrix)
            if -each.radius < o[0] < self.goal_r[0] + each.radius:
                y_upper.append(o[1]+each.radius)
                y_lower.append(o[1]-each.radius)

        for each in g_map.missile:
            o = rotation2st(self.start, each.center, self.rotation_matrix)
            if -each.radius < o[0] < self.goal_r[0] + each.radius:
                y_upper.append(o[1] + each.radius)
                y_lower.append(o[1] - each.radius)
        points = []
        for each in g_map.nfz:
            points.append(rotation2st(self.start, np.array([each.x_min, each.y_min, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_min, each.y_max, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_max, each.y_min, 0]), self.rotation_matrix))
            points.append(rotation2st(self.start, np.array([each.x_max, each.y_max, 0]), self.rotation_matrix))
        y_upper.append(0)
        y_lower.append(0)
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

    def decode_cp(self, _cp):
        distance = np.linalg.norm(self.goal[0:2] - self.start[0:2])
        if _cp == '00':
            len_points = int(distance // 10 + 2)
        elif _cp == '01':
            len_points = int(distance // 5 + 2)
        elif _cp == '10':
            len_points = int(distance // 2 + 2)
        else:
            len_points = int(distance // 1 + 2)
        return len_points


def decode_n_i(_n_i):
    dict_n_i = {'000': 5,
                '001': 10,
                '010': 20,
                '011': 40,
                '100': 100,
                '101': 200,
                '110': 400,
                '111': 600}
    return dict_n_i[_n_i]


def decode_n_p(_n_p):
    dict_n_p = {'00': 1,
                '01': 2,
                '10': 5,
                '11': 10}
    return dict_n_p[_n_p]


def decode_elitism(_elitism):
    disc_elitism = {'00': 0,
                    '01': 0.001,
                    '10': 0.1,
                    '11': 0.2}
    return disc_elitism[_elitism]


def six_bezier(control_points, n):
    return bezier_curve(control_points, n)


def b_spline(control_points, n):
    n -= 1
    xx = [x / n for x in list(range(0, n, 1))]
    degree = 3
    params_list, knots_list = parameterize(control_points[1:-1], degree)
    control_points_x = [each[0] for each in control_points]
    control_points_y = [each[1] for each in control_points]
    control_points_z = [each[2] for each in control_points]
    x = [bspline(a, knots_list, control_points_x, degree) for a in xx]
    y = [bspline(a, knots_list, control_points_y, degree) for a in xx]
    z = [bspline(a, knots_list, control_points_z, degree) for a in xx]

    return np.concatenate((np.array([x, y, z]).T, np.reshape(control_points[-1], (1, 3))), axis=0)


def rts(control_points, n):
    trajectory = []
        # TODO
    return trajectory


def tangent_circle_curve(control_points, n):
    trajectory = []
        # TODO
    return trajectory


def initialize(g_map, num_cp, start, goal_r):
    """
    initialize the control_points
    """
    safe_height = 2
    points = list()
    start_r = [0, 0, start[2]]
    points.append(start_r)
    delta_l = goal_r[0] / (num_cp -1)
    current_terr = g_map.terrain.map(0, 0)
    for i in range(1, num_cp-1):
        temp_cp = list()
        temp_cp.append(delta_l * i)
        temp_cp.append(random.random() * delta_l*2 - delta_l + points[i-1][1])
        if i == 1:
            temp_cp.append(points[0][2] + safe_height)
        else:
            pre_terr = current_terr
            current_terr = g_map.terrain.map(int(temp_cp[0]), int(temp_cp[1]))
            temp_cp.append(np.random.normal(current_terr - pre_terr + points[i-1][2], delta_l / 3))
        points.append(temp_cp)
    points.append(list(goal_r))
    return points


def fitness(trajectory, trajectory_r, population):
    """
    Determine the fitness of an individual. Lower is better.
    """
    len_trajectory = len(trajectory)
    dist_optimal = euclidean_distance(trajectory_r[0], trajectory_r[-1])
    h_safe = 2
    rcs = 1000
    f_3 = 0
    f_4 = 0
    h_1 = 0
    num_out_range = 0
    trajectory_pre = trajectory_r[0:-1]
    trajectory_post = trajectory_r[1:]
    f_1 = sum(euclidean_distance(trajectory_post, trajectory_pre))
    fly_height = trajectory[:, 2] - population.global_map.terrain.map(
                trajectory[:, 0].astype(np.int64), trajectory[:, 1].astype(np.int64))
    f_2 = sum(np.maximum(fly_height, 0)) / len_trajectory

    for radar in population.global_map.radar:
        dis_r = np.minimum(euclidean_distance(radar.center, trajectory) - radar.radius, 0)
        f_3 += sum((-dis_r) / radar.radius * 1 / (1 + 0.1 * (dis_r + radar.radius) ** 4 / rcs))

    for missile in population.global_map.missile:
        dis_m = np.minimum(euclidean_distance(missile.center, trajectory), missile.radius)
        temp_m = missile.radius ** 4
        f_4 += sum((dis_m < missile.radius) * temp_m / (temp_m + dis_m ** 4))

    trajectory_mid = trajectory[int(0.1 * len_trajectory)
                                : int(0.9 * len_trajectory)]
    trajectory_pre_mid = trajectory_mid[0: -1]
    trajectory_next_mid = trajectory_mid[1:]
    line = trajectory_next_mid - trajectory_pre_mid
    line_pre = line[0:-1]
    line_post = line[1:]
    f_5 = sum(np.arccos(np.einsum('ji,ji->j', line_pre, line_post) /
              (np.linalg.norm(line_pre, axis=1) * np.linalg.norm(line_post, axis=1))))
    alpha_k = -1.5377 * 10 ** -10 * trajectory_pre_mid[:, 2] ** 2 \
              - 2.6997 * 10 ** -5 * trajectory_pre_mid[:, 2] + 0.4211
    beta_k = 2.5063 * 10 ** -9 * trajectory_pre_mid[:, 2] ** 2 - \
             6.3014 * 10 ** -6 * trajectory_pre_mid[:, 2] - 0.3257
    s_k = line[:, 2] / (np.sqrt(line[:, 0] ** 2 + line[:, 1] ** 2))

    g_1 = max(max(s_k-alpha_k), 0)
    g_2 = max(max(beta_k-s_k), 0)
    g_3 = max(h_safe - min(fly_height), 0)
    for no_fly in population.global_map.nfz:
        x = np.where((trajectory[:, 0] > no_fly.x_min) & (trajectory[:, 0] < no_fly.x_max))[0]
        y = np.where((trajectory[:, 1] > no_fly.y_min) & (trajectory[:, 1] < no_fly.y_max))[0]
        count = find_same_num(x, y)
        if count:
            h_1 += count
    h_2 = np.sum((trajectory[:, 1] > population.upper_bound) + (trajectory[:, 1] < population.lower_bound))
    return 0.2 * f_1 + 0.1 * f_2 + 0.3 * f_3 + 0.3 * f_4 + 0.1 * f_5,\
           [f_1, f_2, f_3, f_4, f_5], [g_1, g_2, g_3, h_1, h_2], None


def find_same_num(arr_1, arr_2):
    i = j = 0
    cnt = 0
    while i <= len(arr_1)-1 and j <= len(arr_2)-1:
        if arr_1[i] == arr_2[j]:
            cnt += 1

        if arr_1[i] <= arr_2[j]:
            i += 1
        else:
            j += 1
    return cnt


""" old fitness, delete """
# def fitness(trajectory, trajectory_r, population):
#     """
#     Determine the fitness of an individual. Lower is better.
#     """
#     previous_point = trajectory_r[0]
#     next_point = trajectory_r[1]
#     start = trajectory_r[0]
#     goal = trajectory_r[-1]
#     dist_optimal = euclidean_distance(start, goal)
#     h_safe = 2
#     rcs = 1000
#     f_1 = 0
#     f_2 = 0
#     f_3 = 0
#     f_4 = 0
#     f_5 = 0
#     g_1 = 0
#     g_2 = 0
#     g_3 = 0
#     num_nfz = 0
#     num_out_range = 0
#     for i in range(len(trajectory_r)):
#         point = trajectory_r[i]
#         if i < len(trajectory_r) - 1:
#             next_point = trajectory_r[i + 1]
#         f_1 += euclidean_distance(point, previous_point)
#         h = point[2] - population.global_map.terrain.map(int(point[0]), int(point[1]))
#         if h > 0:
#             f_2 += h / len(trajectory_r)
#
#         for radar in population.global_map.radar:
#             dis_r = euclidean_distance(rotation2st(population.start, radar.center, population.rotation_matrix), point)
#             if dis_r < radar.radius:
#                 f_3 += (radar.radius - dis_r) / radar.radius * 1 / (1 + 0.1 * dis_r ** 4 / rcs)
#
#         for missile in population.global_map.missile:
#             dis_m = euclidean_distance(rotation2st(population.start, missile.center, population.rotation_matrix), point)
#             if dis_m < missile.radius:
#                 f_4 += (missile.radius ** 4) / (missile.radius ** 4 + dis_m ** 4)
#
#         if len(trajectory_r) * 0.1 < i < len(trajectory_r) * 0.9:
#             f_5 += np.arccos(np.dot(np.array([point[0] - previous_point[0], point[1] - previous_point[1]]),
#                                     np.array([next_point[0] - point[0], next_point[1] - point[1]]).T) /
#                              (np.linalg.norm(np.array([point[0] - previous_point[0], point[1] - previous_point[1]])) *
#                               np.linalg.norm(np.array([next_point[0] - point[0], next_point[1] - point[1]]))))
#
#             alpha_k = -1.5377 * 10 ** (-10) * point[2] ** 2 - 2.6997 * 10 ** -5 * point[2] + 0.4211
#             beta_k = 2.5063 * 10 ** (-9) * point[2] ** 2 - 6.3014 * 10 ** -6 * point[2] - 0.3257
#             s_k = (next_point[2] - point[2]) / (sqrt((next_point[0] - point[0]) ** 2 + (next_point[1] - point[1]) ** 2))
#             if s_k - alpha_k > g_1:
#                 g_1 = s_k - alpha_k
#
#             if beta_k - s_k > g_2:
#                 g_2 = beta_k - s_k
#             dis_h = h_safe - point[2] + population.global_map.terrain.map(int(point[0]), int(point[1]))
#         else:
#             dis_h = 0
#
#         if dis_h > g_3:
#             g_3 = dis_h
#         for no_fly in population.global_map.nfz:
#             if no_fly.x_min < point[0] < no_fly.x_max and no_fly.y_min < point[1] < no_fly.y_max:
#                 num_nfz += 1
#
#         if point[1] > population.upper_bound or point[1] < population.lower_bound:
#             num_out_range += 1
#
#         previous_point = point
#
#     f_1 = f_1 / dist_optimal
#     h_1 = num_nfz
#     h_2 = num_out_range
#     return 0.2 * f_1 + 0.1 * f_2 + 0.3 * f_3 + 0.3 * f_4 + 0.1 * f_5, [f_1, f_2, f_3, f_4, f_5], [g_1, g_2, g_3, h_1, h_2], None


def binary_search(number, array_input):
    length_array = len(array_input)
    if length_array == 1:
        return 0
    length_array = int(length_array/2)
    if array_input[length_array - 1] <= number <= array_input[length_array]:
        return length_array
    if array_input[length_array - 1] <= number and array_input[length_array] <= number:
        return binary_search(number, array_input[length_array:]) + length_array
    else:
        return binary_search(number, array_input[0:length_array])


def test():

    temp = terrain.generate_map()[0]
    terr = Terr(temp)
    missile, radar, nfz = mapconstraint.generate_constraint(10, 10, 10, terr.points)
    global_map = Map(terr, missile, radar, nfz)
    t0 = time.clock()
    p = Population(np.array([0, 0, global_map.terrain.map(0, 0)]),
                   np.array([50, 50, global_map.terrain.map(50, 50)]), global_map,
                   '011', curve_type='00', _cp='01', _rank_param='10', _elitism='00')
    print(time.clock() - t0)
    fig = plt.figure()
    ax = Axes3D(fig)
    for data in p.data:
        plt_fig(data.trajectory, ax)
    terrain.plt_terrain(p.start, p.goal, p.global_map, ax)
    fig = plt.figure()
    plt.plot([p.data[i].fitness_wight_factor for i in range(len(p.data))])
    return p


if __name__ == '__main__':
    p = test()
