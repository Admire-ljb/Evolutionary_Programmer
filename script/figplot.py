import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
from calculate import euclidean_distance
from mpl_toolkits.mplot3d import Axes3D
from osgeo import gdal
from mpl_toolkits.mplot3d import axes3d
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "Times New Roman"

line_types = ['-ob', '-^r', '-Pc', '-xk', '-sm', '-dy', '-<g', '-Xr', '-*r', '->r', '-hr', '-2r', '-+r', '-vr',
              '-4r', '-1r', '-3r', '-5r', '-6r', '-7r', '-9r', '-_r']


class Trajectory:
    def __init__(self, points, line_style, label):
        self.points = points
        self.linestyle = line_style
        self.label = label


def plt_3d_trajectories(ax_, trajectories):
    for each in trajectories:
        plt_trajectory(each, ax_)


def plot_circle(center, r, x_limit, y_limit, line_style, cmap, label=None):
    x = np.linspace(center[0] - r, center[0] + r, 500)
    tmp = r**2 - (x-center[0])**2
    tmp = np.maximum(tmp, 0)
    tmp = np.sqrt(tmp)
    y1 = tmp + center[1]
    y2 = -tmp + center[1]
    x1_ = []
    x2_ = []
    y1_ = []
    y2_ = []
    # plt.xlim(x_limit[0], x_limit[1])
    # plt.ylim(y_limit[0], y_limit[1])

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = 10 * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    x_plot = []
    y_plot = []
    z_plot = []
    cnt = -1
    for i in range(x.shape[0]):
        flag = 0
        for j in range(x.shape[1]):
            if x[i][j] < x_limit[0] or x[i][j] > x_limit[1]:
                continue
            if y_limit[0] < y[i][j] < y_limit[1]:
                # if flag == 0:
                    # x_plot.append([])
                    # y_plot.append([])
                    # z_plot.append([])
                    # flag = 1
                    # cnt += 1
                x_plot.append(x[i][j])
                y_plot.append(y[i][j])
                z_plot.append(z[i][j])

    try:
        plt.tricontour(x_plot, y_plot, z_plot, line_style, levels=5, linewidths=1, alpha=0.5, zorder=10)
        plt.tricontourf(x_plot, y_plot, z_plot, levels=5, cmap=cmap, alpha=0.5, zorder=10)
    except ValueError:
        pass
    #plt.contour(x_plot, y_plot, z_plot, line_style, levels=10, linewidths=2)
    #plt.contourf(x_plot, y_plot, z_plot, levels=10, cmap="RdBu_r")
    # plt.contour(x2_, y2_, z2_, line_style, levels=14, linewidths=2)
    # plt.contourf(x2_, y2_, z2_, levels=14, cmap="RdBu_r")


def plot_nfz(nfz, limit_x, limit_y, line_style, label):
    if nfz.x_min > limit_x[1] or nfz.y_min > limit_x[1] or nfz.x_max < limit_x[0] or nfz.y_max < limit_y[0]:
        return 1
    x_min = max(nfz.x_min, limit_x[0])
    x_max = min(nfz.x_max, limit_x[1])
    y_min = max(nfz.y_min, limit_y[0])
    y_max = min(nfz.y_max, limit_y[1])
    x = np.linspace(x_min, x_max, 500)
    y = np.linspace(y_min, y_max, 500)
    plt.plot(x, np.zeros(500)+y_min, 'k', linewidth=1, label=label, alpha=0.5)

    plt.plot(x, np.zeros(500)+y_max, 'k',  linewidth=1, alpha=0.5)
    plt.plot(np.zeros(500)+x_min, y, 'k',  linewidth=1, alpha=0.5)
    plt.plot(np.zeros(500)+x_max, y, 'k',  linewidth=1, alpha=0.5)
    plt.fill_between(x, y_min, y_max, facecolor=line_style, alpha=0.5, zorder=10)
    return 0


def plt_contour(start_point, target_point, g_map, routes):
    # 建立步长为0.01，即每隔0.01取一个点
    fig_1 = plt.figure()
    begin_x, end_x, begin_y, end_y, x, y, z = get_plt_x_y_z(start_point, target_point, g_map)
    plt.contour(x, y, z, 40)

    cnt = 0
    for each in routes:
        points_ = each.points
        l = len(points_)
        x_tmp = [points_[i][0] for i in range(l)]
        y_tmp = [points_[i][1] for i in range(l)]
        if len(each.linestyle) < 3:
            plt.plot(x_tmp, y_tmp, each.linestyle, color='orange', label=each.label)
        else:
            plt.plot(x_tmp, y_tmp, each.linestyle, label=each.label)
        cnt += 1
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=4, mode="expand", borderaxespad=0.)
    flag = 0
    for each in g_map.missile:
        center = each.center
        r = each.radius
        label = None
        # if flag == 0:
        #     label = 'missile'
        plot_circle(center, r, [begin_x, end_x], [begin_y, end_y], ':m', "Greys", label)
        flag = 1
    flag = 0
    for each in g_map.radar:
        center = each.center
        r = each.radius
        label = None
        plot_circle(center, r, [begin_x, end_x], [begin_y, end_y], 'k', 'OrRd', label)
        flag = 1
    flag = 0
    for each in g_map.nfz:
        label = None
        if not plot_nfz(each, [begin_x, end_x], [begin_y, end_y], 'm', label):
            flag = 1
    # plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    plt.show()


def new_trajectory(trajectories, points, label):
    s = Trajectory(points, line_types[len(trajectories)], label)
    trajectories.append(s)


def plt_trajectory(trajectory, ax):
    # 3d fig

    l = len(trajectory.points)
    X = [trajectory.points[i][0] for i in range(l)]
    Y = [trajectory.points[i][1] for i in range(l)]
    Z = [trajectory.points[i][2] for i in range(l)]
    if len(trajectory.linestyle) < 3:
        ax.plot(X, Y, Z, trajectory.linestyle, color='orange', alpha=1, linewidth=1, label=trajectory.label, zorder=20)
    else:
        ax.plot(X, Y, Z, trajectory.linestyle, alpha=1, linewidth=1, label=trajectory.label, zorder=20)

    #ax.plot(X, Y, Z, alpha=1, linewidth=1, label=label, color=color, linestyle=linestyle, marker=marker, zorder=20)
    inx = np.random.randint(0, l)
    #ax.text(X[inx], Y[inx], Z[inx], trajectory.describe, fontsize=15)
    plt.legend()


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
        begin_x = int(start_point[0]) - 10
        end_x = int(target_point[0]) + 10
    else:
        begin_x = int(target_point[0]) - 10
        end_x = int(start_point[0]) + 10
    if start_point[1] < target_point[1]:
        begin_y = int(start_point[1]) - 10
        end_y = int(target_point[1]) + 10
    else:
        begin_y = int(target_point[1]) - 10
        end_y = int(start_point[1]) + 10
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
    for each in range(0, int(z_max), 2):
        z_range.append(each)
    ax_3d.set_xticks(x_range)

    ax_3d.set_yticks(y_range)
    ax_3d.set_zticks(z_range)
    # fm = FontProperties(weight='normal', size=20, family='Times New Roman')
    # ax_3d.set_xticklabels(x_range, fontproperties=fm)
    # ax_3d.set_yticklabels(y_range, fontproperties=fm)
    # ax_3d.set_zticklabels(z_range, fontproperties=fm)
    ax_3d.w_xaxis.pane.set_color('w')
    ax_3d.w_yaxis.pane.set_color('w')
    ax_3d.w_zaxis.pane.set_color('w')
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    # plt.rcParams.update({'font.size': 25})
    plt.grid(color='g')
    # plt_missile(start_point, target_point, g_map, ax_3d)