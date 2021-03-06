import parameter_selection as ps
import BaseFunction as bf
import numpy as np
import curve_fitting_error as cfe
import surface_fitting_error as sfe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_curve_data(filename):
    D_X = []
    D_Y = []
    with open(filename) as file:
        for line in file.readlines():
            line = line.strip()
            word = line.split(' ')
            D_X.append(float(word[0]))
            D_Y.append(float(word[1]))
    return [D_X, D_Y]


def load_surface_data(filename):
    D_X = []
    D_Y = []
    D_Z = []
    with open(filename) as file:
        line = file.readline()
        print(line)
        line = line.strip()
        word = line.split(' ')
        row = int(word[0])
        col = int(word[1])
        for i in range(row):
            D_X_row = []
            D_Y_row = []
            D_Z_row = []
            for j in range(col):
                line = file.readline()
                line = line.strip()
                word = line.split(' ')
                D_X_row.append(float(word[1]))
                D_Y_row.append(float(word[2]))
                D_Z_row.append(float(word[3]))
            D_X.append(D_X_row)
            D_Y.append(D_Y_row)
            D_Z.append(D_Z_row)
    return [D_X, D_Y, D_Z]


def LSPIA_curve():
    '''
    The LSPIA iterative method for blending curves.
    '''
    # D_X = [1, 1, 0, -0.5, 1.5,   3, 4, 4.2, 5, 5.5,  6, 5.1, 4.6]
    # D_Y = [0, 1, 2,    3,   4, 3.5, 3, 2.5, 2, 1.2,  1, 2.7,   4]
    # D = [D_X, D_Y]
    D = load_curve_data('cur_data')
    D_X = D[0]
    D_Y = D[1]
    D_N = len(D_X)
    k = 3  # degree
    P_N = int(D_N / 2) - 50
    print(P_N)
    '''
    Step 1. Calculate the parameters
    '''
    p_centripetal = ps.centripetal(D_N, D)
    print(p_centripetal)

    '''
    Step 2. Calculate knot vector
    '''
    knot_vector = ps.LSPIA_knot_vector(p_centripetal, k, P_N, D_N)
    print(knot_vector)

    '''
    Step 3. Select initial control points
    '''
    P_X = [D_X[0]]
    P_Y = [D_Y[0]]
    for i in range(1, P_N - 1):
        # f_i = int(D_N * i / P_N)
        # P_X.append(D_X[f_i])
        # P_Y.append(D_Y[f_i])
        P_X.append(0)
        P_Y.append(0)
    P_X.append(D_X[-1])
    P_Y.append(D_Y[-1])
    P = [P_X, P_Y]

    '''
    Step 4. Calculate the collocation matrix of the NTP blending basis
    '''
    Nik = np.zeros((D_N, P_N))
    c = np.zeros((1, P_N))
    for i in range(D_N):
        for j in range(P_N):
            Nik[i][j] = bf.BaseFunction(j, k + 1, p_centripetal[i], knot_vector)
            c[0][j] = c[0][j] + Nik[i][j]
    C = max(c[0].tolist())
    miu = 2 / C

    '''
    Step 5. First iteration
    '''
    e = []
    ek = cfe.curve_fitting_error(D, P, Nik)
    e.append(ek)

    '''
    Step 6. Adjusting control points
    '''
    P = cfe.curve_adjusting_control_points(D, P, Nik, miu)
    # print(P)
    ek = cfe.curve_fitting_error(D, P, Nik)
    e.append(ek)

    cnt = 0
    while (abs(e[-1] - e[-2]) >= 1e-7):
        cnt = cnt + 1
        print('iteration ', cnt)
        P = cfe.curve_adjusting_control_points(D, P, Nik, miu)
        # print(P)
        ek = cfe.curve_fitting_error(D, P, Nik)
        e.append(ek)
    # print(P)

    '''
    Step 7. Calculate data points on the b-spline curve
    '''
    piece = 200
    p_piece = np.linspace(0, 1, piece)
    Nik_piece = np.zeros((piece, P_N))
    for i in range(piece):
        for j in range(P_N):
            Nik_piece[i][j] = bf.BaseFunction(j, k + 1, p_piece[i], knot_vector)
    P_piece = cfe.curve(p_piece, P, Nik_piece)

    '''
    Step 8. Draw b-spline curve
    '''
    for i in range(D_N):
        plt.scatter(D[0][i], D[1][i], color='r')
    # for i in range(len(P[0]) - 1):
    #     plt.scatter(P[0][i], P[1][i], color='b')
    # for i in range(len(P[0]) - 1):
    #     tmp_x = [P[0][i], P[0][i + 1]]
    #     tmp_y = [P[1][i], P[1][i + 1]]
    #     plt.plot(tmp_x, tmp_y, color='b')

    for i in range(piece - 1):
        tmp_x = [P_piece[0][i], P_piece[0][i + 1]]
        tmp_y = [P_piece[1][i], P_piece[1][i + 1]]
        plt.plot(tmp_x, tmp_y, color='g')

    plt.show()


def LSPIA_surface():
    '''
    The LSPIA iterative method for blending surfaces.
    '''
    D = load_surface_data('surface_data2')
    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]

    row = len(D_X)
    col = len(D_X[0])

    p = 3  # degree
    q = 3
    P_h = int(row / 2)  # the number of control points
    P_l = int(col / 2)

    '''
    Step 1. Calculate the parameters
    '''
    param_u = []
    tmp_param = np.zeros((1, row))
    for i in range(col):
        D_col_X = [x[i] for x in D_X]
        D_col_Y = [y[i] for y in D_Y]
        D_col_Z = [z[i] for z in D_Z]
        D_col = [D_col_X, D_col_Y, D_col_Z]
        tmp_param = tmp_param + np.array(ps.centripetal(row, D_col))
    param_u = np.divide(tmp_param, col).tolist()[0]

    param_v = []
    tmp_param = np.zeros((1, col))
    for i in range(row):
        D_row_X = D_X[i]
        D_row_Y = D_Y[i]
        D_row_Z = D_Z[i]
        D_row = [D_row_X, D_row_Y, D_row_Z]
        tmp_param = tmp_param + np.array(ps.centripetal(col, D_row))
    param_v = np.divide(tmp_param, row).tolist()[0]

    '''
    Step 2. Calculate the knot vectors
    '''
    knot_uv = [[], []]
    knot_uv[0] = ps.LSPIA_knot_vector(param_u, p, P_h, row)
    knot_uv[1] = ps.LSPIA_knot_vector(param_v, q, P_l, col)

    '''
    Step 3. Select initial control points
    '''
    P_X = []
    P_Y = []
    P_Z = []
    for i in range(0, P_h - 1):
        f_i = int(row * i / P_h)
        P_X_row = []
        P_Y_row = []
        P_Z_row = []
        for j in range(0, P_l - 1):
            # f_j = int(col * j / P_l)
            # P_X_row.append(D_X[f_i][f_j])
            # P_Y_row.append(D_Y[f_i][f_j])
            # P_Z_row.append(D_Z[f_i][f_j])
            P_X_row.append(0)
            P_Y_row.append(0)
            P_Z_row.append(0)
        # P_X_row.append(D_X[f_i][-1])
        # P_Y_row.append(D_Y[f_i][-1])
        # P_Z_row.append(D_Z[f_i][-1])
        P_X_row.append(0)
        P_Y_row.append(0)
        P_Z_row.append(0)
        P_X.append(P_X_row)
        P_Y.append(P_Y_row)
        P_Z.append(P_Z_row)

    P_X_row = []
    P_Y_row = []
    P_Z_row = []
    for j in range(0, P_l - 1):
        f_j = int(col * j / P_l)
        P_X_row.append(D_X[-1][f_j])
        P_Y_row.append(D_Y[-1][f_j])
        P_Z_row.append(D_Z[-1][f_j])
    P_X_row.append(D_X[f_i][-1])
    P_Y_row.append(D_Y[f_i][-1])
    P_Z_row.append(D_Z[f_i][-1])
    P_X.append(P_X_row)
    P_Y.append(P_Y_row)
    P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]

    '''
    Step 4. Calculate the collocation matrix of the NTP blending basis
    '''
    Nik_u = np.zeros((row, P_h))
    c_u = np.zeros((1, P_h))
    for i in range(row):
        for j in range(P_h):
            Nik_u[i][j] = bf.BaseFunction(j, p + 1, param_u[i], knot_uv[0])
            c_u[0][j] = c_u[0][j] + Nik_u[i][j]
    C = max(c_u[0].tolist())
    miu_u = 2 / C

    Nik_v = np.zeros((col, P_l))
    c_v = np.zeros((1, P_l))
    for i in range(col):
        for j in range(P_l):
            Nik_v[i][j] = bf.BaseFunction(j, q + 1, param_v[i], knot_uv[1])
            c_v[0][j] = c_v[0][j] + Nik_v[i][j]
    C = max(c_v[0].tolist())
    miu_v = 2 / C

    # miu = miu_u * miu_v
    miu = 0.2
    Nik = [Nik_u, Nik_v]

    '''
    Step 5. First iteration
    '''
    e = []
    ek = sfe.surface_fitting_error(D, P, Nik)
    e.append(ek)

    '''
    Step 6. Adjusting control points
    '''
    P = sfe.surface_adjusting_control_points(D, P, Nik, miu)
    # print(P)
    ek = sfe.surface_fitting_error(D, P, Nik)
    e.append(ek)

    cnt = 0
    while (abs(e[-1] - e[-2]) >= 1e-7):
        cnt = cnt + 1
        print('iteration ', cnt)
        P = sfe.surface_adjusting_control_points(D, P, Nik, miu)
        # print(P)
        ek = sfe.surface_fitting_error(D, P, Nik)
        e.append(ek)
    print(ek)
    '''
    Step 7. Calculate data points on the b-spline curve
    '''
    piece_u = 50
    piece_v = 50
    p_piece_u = np.linspace(0, 1, piece_u)
    p_piece_v = np.linspace(0, 1, piece_v)
    Nik_piece_u = np.zeros((piece_u, P_h))
    Nik_piece_v = np.zeros((piece_v, P_l))
    for i in range(piece_u):
        for j in range(P_h):
            Nik_piece_u[i][j] = bf.BaseFunction(j, p + 1, p_piece_u[i], knot_uv[0])
    for i in range(piece_v):
        for j in range(P_l):
            Nik_piece_v[i][j] = bf.BaseFunction(j, q + 1, p_piece_v[i], knot_uv[1])

    p_piece = [piece_u, piece_v]
    Nik_piece = [Nik_piece_u, Nik_piece_v]
    P_piece = sfe.surface(p_piece, P, Nik_piece)

    '''
    Step 8. Draw b-spline curve
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(int(row/10)):
        for j in range(int(col/10)):
            ax.scatter(D_X[10*i][10*j], D_Y[10*i][10*j], D_Z[10*i][10*j], color='r')
    # for i in range(len(P[0]) - 1):
    #     plt.scatter(P[0][i], P[1][i], color='b')
    # for i in range(len(P[0]) - 1):
    #     tmp_x = [P[0][i], P[0][i + 1]]
    #     tmp_y = [P[1][i], P[1][i + 1]]
    #     plt.plot(tmp_x, tmp_y, color='b')

    for i in range(piece_u):
        for j in range(piece_v - 1):
            tmp_x = [P_piece[0][i][j], P_piece[0][i][j + 1]]
            tmp_y = [P_piece[1][i][j], P_piece[1][i][j + 1]]
            tmp_z = [P_piece[2][i][j], P_piece[2][i][j + 1]]
            ax.plot(tmp_x, tmp_y, tmp_z, color='g')
    for j in range(piece_v):
        for i in range(piece_u - 1):
            tmp_x = [P_piece[0][i][j], P_piece[0][i + 1][j]]
            tmp_y = [P_piece[1][i][j], P_piece[1][i + 1][j]]
            tmp_z = [P_piece[2][i][j], P_piece[2][i + 1][j]]
            ax.plot(tmp_x, tmp_y, tmp_z, color='g')

    plt.show()


# LSPIA_curve()

LSPIA_surface()
