import random
from calculate import *
# import time
# from decoder import *
from generatemap.terrain import *
from generatemap import mapconstraint
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from osgeo import gdal
from mpl_toolkits.mplot3d import axes3d

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
        self.control_points_r = control_points
        # self.control_points = rotation2origin(pop.start, self.control_points_r, pop.rotation_matrix)
        self.trajectory_r = self.curve(pop.soft_inf.curve_type, pop.len_individual)
        self.trajectory = rotation2origin(pop.start, self.trajectory_r, pop.rotation_matrix)
        self.fitness_wight_factor, self.fitness, self.constraint = \
            fitness(self.trajectory, self.trajectory_r, pop)
        self.sort_basis = None
        # self.sort_basis = None
        self.velocity = np.zeros([pop.num_cp, 3])
        # self.p_best = self

    def curve(self, curve, n):
        """"""
        return eval(curve)(self.control_points_r, n)


def per_20(distance):
    return int(distance // 20 + 2)


def per_10(distance):
    return int(distance // 10 + 2)


def per_5(distance):
    return int(distance // 5 + 2)


def per_2(distance):
    return int(distance // 2 + 2)


def linear_rank(num):
    return np.linspace(num, 1, num)


def sqrt_rank(num):
    return np.sqrt(np.linspace(num, 1, num))


def ex_rank(num):
    return np.linspace(num, 1, num) ** 2


def reciprocal_rank(num):
    return 1 / np.linspace(1, num, num)


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
    return b_spline(control_points, n)


def tangent_circle_curve(control_points, n):
    trajectory = []
        # TODO
    return bezier_curve(control_points, n)


def initialize(g_map, num_cp, start, goal_r, matrix):
    """
    initialize the control_points
    """
    safe_height = 2
    points = list()
    start_r = [0, 0, start[2]]
    points.append(start_r)
    delta_l = goal_r[0] / (num_cp - 1)
    current_terr = g_map.terrain.map(start[0], start[1])
    for i in range(1, num_cp-1):
        temp_cp = list()
        temp_cp.append(delta_l * i)
        temp_cp.append(random.random() * delta_l*2 - delta_l + points[i-1][1])
        if i == 1:
            temp_cp.append(points[0][2] + safe_height)
        else:
            pre_terr = current_terr
            temp_cp_origin = np.dot(np.array([temp_cp[0], temp_cp[1], 0]), matrix) + [start[0], start[1], 0]
            current_terr = g_map.terrain.map(int(temp_cp_origin[0]), int(temp_cp_origin[1]))

            temp_cp.append(np.random.normal(current_terr + max(points[i-1][2] - pre_terr, safe_height), delta_l / 3))
        points.append(temp_cp)
    points.append(list(goal_r))
    return points


def fitness(trajectory, trajectory_r, population):
    """
    Determine the fitness of an individual. Lower is better.
    """
    len_trajectory = len(trajectory)
    trajectory_ = trajectory[len_trajectory // 10: len_trajectory // 10 * 9]
    trajectory_r_ = trajectory_r[len_trajectory // 10: len_trajectory // 10 * 9]
    len_trajectory = len(trajectory_)
    dist_optimal = population.distance
    h_safe = 0.5
    rcs = 1000
    f_3 = 0
    f_4 = 0
    h_1 = 0
    trajectory_pre = trajectory_r_[0:-1]
    trajectory_post = trajectory_r_[1:]
    f_1 = sum(euclidean_distance(trajectory_post, trajectory_pre)) / dist_optimal
    tmp = population.global_map.terrain.map(trajectory_[:, 0].astype(np.int64), trajectory_[:, 1].astype(np.int64))
    if tmp is None:
        fly_height = 100
    else:
        fly_height = trajectory_[:, 2] - tmp
    f_2 = sum(np.maximum(fly_height, 0)) / len_trajectory

    for radar in population.global_map.radar:
        dis_r = np.minimum(euclidean_distance(radar.center, trajectory_) - radar.radius, 0)
        f_3 += sum((-dis_r) / radar.radius * 1 / (1 + 0.1 * (dis_r + radar.radius) ** 4 / rcs))

    for missile in population.global_map.missile:
        dis_m = np.minimum(euclidean_distance(missile.center, trajectory_), missile.radius)
        temp_m = missile.radius ** 4
        f_4 += sum((dis_m < missile.radius) * temp_m / (temp_m + dis_m ** 4))

    trajectory_mid = trajectory_[int(0.1 * len_trajectory)
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
        x = np.where((trajectory_[:, 0] > no_fly.x_min) & (trajectory_[:, 0] < no_fly.x_max))[0]
        y = np.where((trajectory_[:, 1] > no_fly.y_min) & (trajectory_[:, 1] < no_fly.y_max))[0]
        count = find_same_num(x, y)
        if count:
            h_1 += count
    h_2 = np.sum((trajectory_[:, 0:2] > population.upper_bound) + (trajectory_[:, 0:2] < population.lower_bound))

    return 0.2 * f_1 + 0.1 * f_2 + 0.3 * f_3 + 0.3 * f_4 + 0.1 * f_5, \
           np.array([f_1, f_2, f_3, f_4, f_5]), np.array([g_1, g_2, g_3, h_1, h_2])


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
#     return 0.2 * f_1 + 0.1 * f_2 + 0.3 * f_3 + 0.3 * f_4 + 0.1 * f_5, [f_1, f_2, f_3, f_4, f_5], [g_1, g_2, g_3, h_1, h_2]


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


def in_zone(point_, start, end):
    if start[0] > point_[0] > end[0] or start[0] < point_[0] < end[0]:
        if start[1] > point_[1] > end[1] or start[1] < point_[1] < end[1]:
            return True
    return False


def plt_missile(start_point, target_point, g_map, ax_3d):
    # Grab some test data.
    # g_map.missle
    # x = np.arange(begin_x, end_x, 1)
    # y = np.arange(begin_y, end_y, 1)
    # x, y = np.meshgrid(x, y)
    # z = g_map.terrain.map(x, y)
    # z_max = z.max()
    start = np.array([start_point[0], start_point[1]])
    end = np.array([target_point[0], target_point[1]])
    for each in g_map.missile:
        center = np.array(each.center[0:2])
        if in_zone(center, start, end):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi/2, 100)
            x = 10 * np.outer(np.cos(u), np.sin(v)) + each.center[0]
            y = 10 * np.outer(np.sin(u), np.sin(v)) + each.center[1]
            z = 10 * np.outer(np.ones(np.size(u)), np.cos(v)) + each.center[2]
            ax_3d.plot_wireframe(x, y, z, rstride=10, cstride=10, zorder=10, color='c')
    plt.show()
    return

def get_plt_x_y_z(start_point, target_point, g_map):
    if start_point[0] < target_point[0]:
        begin_x = int(start_point[0]) - 2
        end_x = int(target_point[0]) + 2
    else:
        begin_x = int(target_point[0]) - 2
        end_x = int(start_point[0]) + 2
    if start_point[1] < target_point[1]:
        begin_y = int(start_point[1]) - 2
        end_y = int(target_point[1]) + 2
    else:
        begin_y = int(target_point[1]) - 2
        end_y = int(start_point[1]) + 2
    x = np.arange(begin_x, end_x, 1)
    y = np.arange(begin_y, end_y, 1)
    x, y = np.meshgrid(x, y)
    z = g_map.terrain.map(x, y)
    return begin_x, end_x, begin_y, end_y, x, y, z


def plt_terrain(start_point, target_point, g_map, ax_3d):
    begin_x, end_x, begin_y, end_y, x, y, z = get_plt_x_y_z(start_point, target_point, g_map)
    z_max = z.max()
    # ax_3d.plot_surface(x, y, z,
    #                    rstride=2, cstride=2,
    #                    cmap=plt.get_cmap('rainbow'),
    #                    alpha=1,
    #                    edgecolors=[0, 0, 0])
    # ax_3d.plot_surface(x, y, z,
    #                    rstride=2, cstride=2,
    #                    cmap=plt.get_cmap('rainbow'),
    #                    alpha=0.1,
    #                    edgecolors=[0, 0, 0])

    ls = LightSource(270, 20)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax_3d.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=1, antialiased=False, shade=False, zorder=0)
   # ax_3d.contour(x, y, z, zdim='z', offset=-2, cmap = 'rainbow')
    # cset = ax_3d.contour(x, y, z, zdir='y', offset=3, cmap='binary')
    # cset = ax_3d.contour(x, y, z, zdir='x', offset=-3, cmap='Blues')
    x_range = []
    y_range = []
    z_range = []
    for each in range(int(begin_x/10) * 10 + 10, end_x, 10):
        x_range.append(each)
    for each in range(int(begin_y/ 10) * 10 + 10, end_y, 10):
        y_range.append(each)
    for each in range(0, int(z_max), 10):
        z_range.append(each)
    ax_3d.set_xticks(x_range)
    ax_3d.set_yticks(y_range)
    ax_3d.set_zticks(z_range)
    plt_missile(start_point, target_point, g_map, ax_3d)


def test_map():
    temp_ = generate_map()[0]
    terra = mapconstraint.Terr(temp_)
    missile_, radar_, nfz_ = mapconstraint.generate_constraint(0, 0, 0, terra.points)
    g_map = mapconstraint.Map(terra, missile_, radar_, nfz_)
    return g_map


def generate_map_in_constrain(global_map, num_missile, num_radar, num_nfz):
    missile_, radar_, nfz_ = mapconstraint.generate_constraint(num_missile, num_radar, num_nfz, global_map.terrain)
    return mapconstraint.Map(global_map.terrain, missile_, radar_, nfz_)

