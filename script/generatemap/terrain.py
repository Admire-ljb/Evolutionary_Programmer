import graphics.engine
import perlin
import math
import pcl
import numpy as np
from pcl import pcl_visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
from matplotlib import cm

############ Display variables

scale = 6
distance = 100

############ Land size

width = 150 # map width
length = 100 # map length

############ Noise variables

n1div = 30 # landmass distribution
n2div = 4 # boulder distribution
n3div = 1 # rock distribution

n1scale = 20 # landmass height
n2scale = 2 # boulder scale
n3scale = 0.5 # rock scale

noise1 = perlin.noise(width / n1div, length / n1div) # landmass / mountains
noise2 = perlin.noise(width / n2div, length / n2div) # boulders
noise3 = perlin.noise(width / n3div, length / n3div) # rocks

############ z modifiers

zroot = 2
zpower = 2.5


def color(a, b, c): # check land type
    ############ colors
    colors = {
        0: 'blue',
        1: 'yellow',
        20: 'green',
        25: 'gray',
        1000: 'white'
    }
    z = (points[a][2] + points[b][2] + points[c][2]) / 3 # calculate average height of triangle
    for color in colors:
        if z <= color:
            return colors[color]


def generate_map():
    ############ 3D shapes
    points = []

    point = []
    ############
    for x in range(-int(width/2), int(width/2)):
        for y in range(-int(length/2), int(length/2)):
            x1 = x + width/2
            y1 = y + length/2
            z = noise1.perlin(x1 / n1div, y1 / n1div) * n1scale # add landmass
            z += noise2.perlin(x1 / n2div, y1 / n2div) * n2scale # add boulders
            z += noise3.perlin(x1 / n3div, y1 / n3div) * n3scale # add rocks
            if z >= 0:
                z = -math.sqrt(z)
            else:
                z = ((-z) ** (1 / zroot)) ** zpower
            points.append([x, y, z])
            point.append([x+75, y+50, z])
    Z = [points[i][2] for i in range(15000)]
    Z = np.array(Z)
    Z_min = Z.min()
    for i in range(15000):
        point[i][2] -= Z_min
    return point, points


def plt_fig(points, ax, missile=[], radar=[], nfz=[]):
    # 3d fig
    X =[points[i][0] for i in range(15000)]
    Y =[points[i][1] for i in range(15000)]
    Z =[points[i][2] for i in range(15000)]

    X = np.array(X)
    X = X.reshape(150, 100)
    Y = np.array(Y)
    Y = Y.reshape(150, 100)
    Z = np.array(Z)
    Z = Z.reshape(150, 100)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * (Y.max() - Y.min()) * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    '''cmap是颜色映射表
    from matplotlib import cm
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
    cmap = "rainbow" 亦可
    我的理解的 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
    你也可以修改 rainbow 为 coolwarm, 验证我的结论'''
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.switch_backend('agg')
    for i in range(len(missile)):
        # ax2 = fig.gca(projection='3d')
        # ax2.set_aspect("equal")

        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = missile[i].radius * np.cos(u) * np.sin(v) + missile[i].center[0]
        y = missile[i].radius * np.sin(u) * np.sin(v) + missile[i].center[1]
        z = np.cos(v) + missile[i].center[2]
        ax.plot_wireframe(x, y, z, color="r")
    # ax.set_zlim(Z.min()-10, Z.max() + 10)


def constraint_plt(missile, radar, nfz, fig):
    ax = fig.gca(projection='3d')
    for i in range(len(missile)):
        # draw sphere
        u, v = np.mgrid[0: 2*np.pi:40j, 0:1/2 * np.pi:40j]
        x = missile[i].radius * np.cos(u) * np.sin(v) + missile[i].center[0]
        y = missile[i].radius * np.sin(u) * np.sin(v) + missile[i].center[1]
        z = missile[i].radius * np.cos(v) + missile[i].center[2]
        ax.plot_surface(x, y, z, color="r")
        # # Create cubic bounding box to simulate equal aspect ratio
        # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        # # Comment or uncomment following both lines to test the fake bounding box:
        # for xb, yb, zb in zip(Xb, Yb, Zb):
        #     ax.plot([xb], [yb], [zb], 'w')


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
    z = g_map.terrain.map(x, y)
    ax_3d.plot_surface(x, y, z,
                       rstride=2, cstride=2,
                       cmap=plt.get_cmap('rainbow'),
                       alpha=1,
                       edgecolors=[0, 0, 0])
    # ax_3d.contour(x, y, z, zdir='z', offset=-1, camp='rainbow')


if __name__ == "__main__":
    triangles = []
    point, points = generate_map()
    for x in range(width):
        for y in range(length):
            if 0 < x and 0 < y:
                a, b, c = int(x * length + y), int(x * length + y - 1), int((x - 1) * length + y) # find 3 points in triangle
                triangles.append([a, b, c, color(a, b, c)])

            if x < width - 1 and y < length - 1:
                a, b, c, = int(x * length + y), int(x * length + y + 1), int((x + 1) * length + y) # find 3 points in triangle
                triangles.append([a, b, c, color(a, b, c)])

    ############

    world = graphics.engine.Engine3D(points, triangles, scale=scale, distance=distance, width=1400, height=750, title='Terrain')

    world.rotate('x', -30)

    p = pcl.PointCloud(np.array(points, dtype=np.float32))

    viewer = pcl.pcl_visualization.PCLVisualizering(b'cloud')
    viewer.AddPointCloud(p)
    v = True
    while v:
        v = not (viewer.WasStopped())
        viewer.SpinOnce()

    world.render()
    world.screen.window.mainloop()
    # fig define!
    fig = plt.figure()
    ax = Axes3D()
    plt_fig(point, ax)

