from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

AMOUNT_OF_POINTS = 100
AMOUNT_OF_CONTROL = 30


def insertValuesIntoMatrix(matrix, value, row, index, timesToInsert):
    if matrix.size / 2 >= index + timesToInsert:
        for i in range(timesToInsert):
            matrix[row, index + i] = value


def createDeJongF5Matrix():
    a = np.zeros((2, AMOUNT_OF_CONTROL))
    # a.resize(2, AMOUNT_OF_CONTROL)
    for i in range(2):
        value = -32
        for j in range(AMOUNT_OF_CONTROL):
            a[0, j] = value
            value = value + 16
            if j > 0 and (j + 1) % 5 == 0:
                value = -32
                valueIndex = ((j + 1) // 5) - 1
                startIndex = valueIndex * 5
                insertValuesIntoMatrix(a, a[0, valueIndex], 1, startIndex, 5)
    return a


def DeJongF5(x, y, c, matrix):
    sumi = 0
    for i in range(30):
        sumj = (x - matrix[0, i]) ** 2 + (y - matrix[1, i]) ** 2
        sumi += ((c[i] + sumj) ** -1)
    sumi *= 0.1
    return sumi


if __name__ == "__main__":
    # fig define!
    fig = plt.figure()
    # 3d fig
    ax = Axes3D(fig)

    # fig
    X = np.linspace(-25, 40, AMOUNT_OF_POINTS)
    Y = np.linspace(-40, 32, AMOUNT_OF_POINTS)

    # generate mesh data
    X, Y = np.meshgrid(X, Y)

    dejongList = []
    # deJongMatrix = [[9.681, 0.667], [9.400, 2.041],[8.025,9.152],[2.196,0.415],[8.074,8.777],[7.650,5.658],
    #    [1.256,3.605], [8.314,2.261], [0.226, 8.858], [7.305, 2.228], [0.652, 7.027], [2.699,3.516],
    #    [8.327, 3.897],[2.132,7.006],[4.707,5.579],[8.304,7.559],[8.632,4.409],[4.887,9.112],[2.440,6.686],
    #    [6.306, 8.583], [0.652, 2.343], [5.558, 1.272], [3.352, 7.549], [8.798, 0.880], [1.460,8.057],[0.432,8.645],
    #    [0.679, 2.800], [4.263, 1.074], [9.496, 4.830], [4.138, 2.562]]

    # deJongMatrix = np.array(deJongMatrix)
    # deJongMatrix = deJongMatrix.reshape(2, AMOUNT_OF_CONTROL)
    deJongMatrix = createDeJongF5Matrix()
    # C = [0.806, 0.517, 0.100, 0.908,0.965,0.669,0.524,0.902,0.531,0.876,0.462,0.491,0.463,0.714,0.352,0.869
    #      ,0.813, 0.811,0.828,0.964,0.789,0.360,0.369,0.992,0.332,0.817,0.632,0.883,0.608,0.326]

    C = []
    for i in range(30):
        C.append(i+1)
    for i in range(AMOUNT_OF_POINTS):
        val = DeJongF5(X[i], Y[i], C, deJongMatrix)
        dejongList.append(val)
    Z = np.array(dejongList)
    Z *= 100
    X += 25
    Y += 40
    Y = Y
    '''cmap是颜色映射表
    from matplotlib import cm
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
    cmap = "rainbow" 亦可
   我的理解的 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
   你也可以修改 rainbow 为 coolwarm, 验证我的结论'''

    surf = ax.plot_surface(X, -Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"),
                           linewidth=0, antialiased=False)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.switch_backend('TkAgg')
    plt.show()
