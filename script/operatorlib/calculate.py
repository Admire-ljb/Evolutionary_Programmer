import numpy as np
from scipy.special import comb
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
# import matplotlib.pyplot as plt


def euclidean_distance(d1, d2):
    """calculates the distance between two data points
    :param d1: points 1
    :param d2: points 2
    :return: distance between two points
    """
    if d1.ndim == 1 and d2.ndim == 1:
        result = np.linalg.norm(d2 - d1)
    else:
        result = np.linalg.norm(d2-d1, axis=1)
    return result


def rotation2st(start, point, rotation_matrix):
    return np.dot((point - [start[0], start[1], 0]), np.linalg.inv(rotation_matrix))


def rotation2origin(start, point, rotation_matrix):
    return np.dot(point, rotation_matrix) + [start[0], start[1], 0]


def separation_coordinates(points):
    x_points = np.array([p[0] for p in points])
    y_points = np.array([p[1] for p in points])
    z_points = np.array([p[2] for p in points])
    return x_points, y_points, z_points


def bezier_curve(points, nTimes=80):
    """
   Given a set of control points, return the
   bezier curve defined by the control points.
    """
    length_cp = len(points)
    # Creating random points for control points
    x_points, y_points, z_points = separation_coordinates(points)

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, length_cp - 1, t) for i in range(0, length_cp)])
    x_values = np.dot(x_points, polynomial_array)
    y_values = np.dot(y_points, polynomial_array)
    z_values = np.dot(z_points, polynomial_array)

    path = [[x_values[i], y_values[i], z_values[i]] for i in range(0, len(x_values))]
    path.reverse()
    '''
    plt.plot(xvals, yvals)
    plt.plot(xPoints, yPoints, "ro")
    for nr in range(len(xPoints)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.show()
    '''
    path = np.array(path)
    return path


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def set_knots(param_list, degree=3):
    """sets the B-spline knots
    param param_list:
    param degree:
    return:
    """
    t0 = [0.] * degree
    tn = [1.] * degree
    knots_list = t0 + param_list + tn
    return knots_list


def evaluate(t, u, i, j):
    """evaluates the element of the N basis matrix
    :param t:
    :param u:
    :param i:
    :param j:
    :return: N value
    """
    val = 0.

    if u[j] <= t[i] <= u[j + 1] and (t[i] != u[j] or t[i] != u[j + 1]):
        try:
            val = (t[i] - u[j]) ** 3 / ((u[j + 1] - u[j]) * (u[j + 2] - u[j]) * (u[j + 3] - u[j]))
        except ZeroDivisionError:
            val = 0.

    elif u[j + 1] <= t[i] < u[j + 2]:
        try:
            val = ((t[i] - u[j]) ** 2 * (u[j + 2] - t[i])) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 3] - u[j]) * (u[j + 2] - u[j])) + \
                  ((u[j + 3] - t[i]) * (t[i] - u[j]) * (t[i] - u[j + 1])) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 3] - u[j + 1]) * (u[j + 3] - u[j])) + \
                  ((u[j + 4] - t[i]) * ((t[i] - u[j + 1]) ** 2)) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 4] - u[j + 1]) * (u[j + 3] - u[j + 1]))
        except ZeroDivisionError:
            val = 0.

    elif u[j + 2] <= t[i] < u[j + 3]:
        try:
            val = ((t[i] - u[j]) * (u[j + 3] - t[i]) ** 2) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 3] - u[j + 1]) * (u[j + 3] - u[j])) + \
                  ((u[j + 4] - t[i]) * (u[j + 3] - t[i]) * (t[i] - u[j + 1])) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 4] - u[j + 1]) * (u[j + 3] - u[j + 1])) + \
                  ((u[j + 4] - t[i]) ** 2 * (t[i] - u[j + 2])) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 4] - u[j + 2]) * (u[j + 4] - u[j + 1]))
        except ZeroDivisionError:
            val = 0.

    elif u[j + 3] <= t[i] <= u[j + 4] and (t[i] != u[j + 3] or t[i] != u[j + 4]):
        try:
            val = (u[j + 4] - t[i]) ** 3 / (
                    (u[j + 4] - u[j + 3]) * (u[j + 4] - u[j + 2]) * (u[j + 4] - u[j + 1]))
        except ZeroDivisionError:
            val = 0.

    return val


def endpoints(t, u, i, j):
    """endpoints conditions
    :param t:
    :param u:
    :param i:
    :param j:
    :return:
    """
    val_ = 0.

    if u[j] <= t[i] <= u[j + 1] and (t[i] != u[j] or t[i] != u[j + 1]):
        try:
            val_ = 6 * (t[i] - u[j]) / ((u[j + 1] - u[j]) * (u[j + 2] - u[j]) * (u[j + 3] - u[j]))
        except ZeroDivisionError:
            val_ = 0.

    elif u[j + 1] <= t[i] <= u[j + 2] and (t[i] != u[j + 1] or t[i] != u[j + 2]):
        try:
            val_ = (2 * (u[j + 2] - t[i]) - 4 * (t[i] - u[j])) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 3] - u[j]) * (u[j + 2] - u[j])) + \
                   (2 * u[j] - 6 * t[i] + 2 * u[j + 1] + 2 * u[j + 3]) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 3] - u[j + 1]) * (u[j + 3] - u[j])) + \
                   (4 * u[j + 1] - 6 * t[i] + 2 * u[j + 4]) / (
                    (u[j + 2] - u[j + 1]) * (u[j + 4] - u[j + 1]) * (u[j + 3] - u[j + 1]))
        except ZeroDivisionError:
            val_ = 0.

    elif u[j + 2] <= t[i] <= u[j + 3] and (t[i] != u[j + 2] or t[i] != u[j + 3]):
        try:
            val_ = (6 * t[i] - 2 * u[j] - 4 * u[j + 3]) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 3] - u[j + 1]) * (u[j + 3] - u[j])) + \
                   (6 * t[i] - 2 * u[j + 1] - 2 * u[j + 3] - 2 * u[j + 4]) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 4] - u[j + 1]) * (u[j + 3] - u[j + 1])) + \
                   (6 * t[i] - 2 * u[j + 2] - 4 * u[j + 4]) / (
                    (u[j + 3] - u[j + 2]) * (u[j + 4] - u[j + 2]) * (u[j + 4] - u[j + 1]))
        except ZeroDivisionError:
            val_ = 0.

    elif u[j + 3] <= t[i] <= u[j + 4] and (t[i] != u[j + 3] or t[i] != u[j + 4]):
        try:
            val_ = 6 * (u[j + 4] - t[i]) / (
                    (u[j + 4] - u[j + 3]) * (u[j + 4] - u[j + 2]) * (u[j + 4] - u[j + 1]))
        except ZeroDivisionError:
            val_ = 0.

    return val_


def tridiag_solver(a, b, c, d):
    """ Tri-diagonal matrix solver, a b c d can be NumPy array type or Python list type.
    refer to https://blog.csdn.net/weixin_30832351/article/details/97646699
    and to https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    :param a: lower diagonal
    :param b: main diagonal
    :param c: upper diagnoal
    :param d: right hand side of the system
    :return: solution of the system
    """
    nf = len(d)
    ac, bc, cc, dc = map(list, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


def parameterize(data_points, degree, type_='chord'):
    """assigns appropriate parameter values to data points
    :param data_points:
    :param degree:
    :param type_:
    :return:
    """
    # print("{} parameterization of data points".format(type_))

    n = len(data_points)
    t = [0.] * n

    if type_ == 'chord':
        for i in range(1, n):
            numerator = 0
            denominator = 0
            for k in range(1, i + 1):
                numerator += euclidean_distance(data_points[k], data_points[k - 1])
            for k in range(1, n):
                denominator += euclidean_distance(data_points[k], data_points[k - 1])
            t[i] = numerator / denominator

    elif type_ == 'uniform':
        for i in range(1, n):
            t[i] = i / n

    else:
        msg = "Parameterization method doesn't exist"
        raise LookupError(msg)

    t[-1] = 1
    k = set_knots(t, degree)

    return t, k


def basis(params_list, knots_list):
    """Cubic B-spline basis
    :param params_list:
    :param knots_list:
    :return: basis matrix
    """
    print("Calculate B-Spline basis matrix")

    n = len(params_list) - 1
    cnt_num = n + 3  # control points
    basis_mat = np.zeros([cnt_num, cnt_num])

    for i in range(n + 1):
        for j in range(n + 3):
            basis_mat[i + 1][j] = evaluate(params_list, knots_list, i, j)
    for i in range(2):
        for j in range(n + 3):
            basis_mat[i * (n + 2)][j] = endpoints(params_list, knots_list, i * n, j)

    return basis_mat


def solver(basis_mat, data_points):
    """ solve the linear system
    :param basis_mat: basis matrix
    :param data_points: given data points
    :return: a list of control points of the B-Spline
    """
    control_points = []
    n = len(basis_mat[0])
    d0 = np.array([0, 0, 0]).reshape(1, 3)
    appended_data_points = np.concatenate((d0, data_points, d0), axis=0)
    x = [each[0] for each in appended_data_points]
    y = [each[1] for each in appended_data_points]
    z = [each[2] for each in appended_data_points]

    # swap the 1st and 2nd rows, the n - 1 and n rows
    basis_mat[0], basis_mat[1] = basis_mat[1], basis_mat[0]
    basis_mat[n - 2], basis_mat[n - 1] = basis_mat[n - 1], basis_mat[n - 2]
    x[0], x[1] = x[1], x[0]
    x[n - 2], x[n - 1] = x[n - 1], x[n - 2]
    y[0], y[1] = y[1], y[0]
    y[n - 2], y[n - 1] = y[n - 1], y[n - 2]

    # extract diagonal
    lower_diag = [basis_mat[i + 1][i] for i in range(n - 1)]
    main_diag = [basis_mat[i][i] for i in range(n)]
    upper_diag = [basis_mat[i][i + 1] for i in range(n - 1)]

    x_control = tridiag_solver(lower_diag, main_diag, upper_diag, x)
    y_control = tridiag_solver(lower_diag, main_diag, upper_diag, y)

    print("Solve tri-diagnoal linear system")

    for i in range(n):
        control_points.append((x_control[i], y_control[i]))

    return control_points


def B(x, k, i, t):
    """recursive definition of B-Spline curve
    :param x:
    :param k:
    :param i:
    :param t:
    :return:
    """
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    if t[i + k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i]) / (t[i + k] - t[i]) * B(x, k - 1, i, t)
    if t[i + k + 1] == t[i + 1]:
        c2 = 0.0
    else:
        c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B(x, k - 1, i + 1, t)
    return c1 + c2


def bspline(x, t, c, k):
    """evaluate B-Spline curve
    :param x:
    :param t:
    :param c:
    :param k:
    :return:
    """
    n = len(t) - k - 1
    assert (n >= k + 1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))


def zeros_matrix(rows, cols):
    """creates a matrix filled with zeros.
    :param rows: the number of rows the matrix should have
    :param cols: the number of columns the matrix should have
    :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def node_vector(n, k):
    vector = np.zeros(n + k + 2)
    if n % k == 0:
        vector[n + 1: n + k + 2] = 1
        piecewise = int(n / k)
        flag = 0
        if piecewise > 1:
            for i in range(1, piecewise):
                for j in range(1, k+1):
                    vector[k + flag * k + j] = i / piecewise
                flag += 1
    return vector


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


def rotate_coordinate(start, goal):
    theta = np.arctan((goal[1] - start[1])/(goal[0]-start[0]))
    if goal[0] < start[0]:
        theta += np.pi
    if goal[0] == start[0]:
        theta += np.pi * bool(start[1] > goal[1])
    rotation_matrix = np.array([[np.cos(theta),  np.sin(theta), 0],
                                [- np.sin(theta), np.cos(theta), 0],
                                [0,                0,            1]])
    return rotation_matrix


def y_boundary(population, g_map):
    delta_d = 5
    y_upper = []
    y_lower = []
    for each in g_map.radar:
        o = rotation2st(population.start, each.center, population.rotation_matrix)
        if -each.radius < o[0] < population.goal_r[0] + each.radius:
            y_upper.append(o[1]+each.radius)
            y_lower.append(o[1]-each.radius)

    for each in g_map.missile:
        o = rotation2st(population.start, each.center, population.rotation_matrix)
        if -each.radius < o[0] < population.goal_r[0] + each.radius:
            y_upper.append(o[1] + each.radius)
            y_lower.append(o[1] - each.radius)
    points = []
    for each in g_map.nfz:
        points.append(rotation2st(population.start, np.array([each.x_min, each.y_min, 0]), population.rotation_matrix))
        points.append(rotation2st(population.start, np.array([each.x_min, each.y_max, 0]), population.rotation_matrix))
        points.append(rotation2st(population.start, np.array([each.x_max, each.y_min, 0]), population.rotation_matrix))
        points.append(rotation2st(population.start, np.array([each.x_max, each.y_max, 0]), population.rotation_matrix))
    y_upper.append(0)
    y_lower.append(0)
    for i in points:
        if 0 < i[0] < population.goal_r[0]:
            y_upper.append(i[1])
            y_lower.append(i[1])
    return max(max(y_upper), 0) + delta_d, min(min(y_lower), 0) - delta_d


def plt_trajectory(points, ax, label=None, color=None, linestyle=None, marker=None, describe=None):
    # 3d fig
    l = len(points)
    X =[points[i][0] for i in range(l)]
    Y =[points[i][1] for i in range(l)]
    Z =[points[i][2] for i in range(l)]
    ax.plot(X, Y, Z, alpha=1, linewidth=1, label=label, color=color, linestyle=linestyle, marker=marker, zorder=20)
    inx = np.random.randint(0, l)
    ax.text(X[inx], Y[inx], Z[inx], describe, fontsize=15)
