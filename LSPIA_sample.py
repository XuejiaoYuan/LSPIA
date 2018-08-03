"""
    本文件中的method适用场景：
        1.大镜场，如300x300；
        2.用于LSPIA的数据点需要先从300x300的数据点中采样；
        3.利用采样的数据点进行曲面拟合，并计算与真值之间的误差；
    本文思路：
        大镜场计算每一面定日镜的阴影与遮挡耗时过大，考虑采样其中
        部分镜子进行阴影和遮挡计算，通过采样数据拟合曲面，希望能
        与实际的数据贴近。
"""

import parameter_selection as ps
import BaseFunction as bf
import numpy as np
import curve_fitting_error as cfe
import surface_fitting_error as sfe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LSPIA import load_shadow_block_data


def sample_surface_data(D, P_h, P_l):
    '''
    Sample the surface data
    :param D: the surface data
    :param P_h: the number of sample rows
    :param P_l: the number of sample cols
    :return: sample data
    '''
    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]
    row = len(D_X)
    col = len(D_X[0])

    P_X = []
    P_Y = []
    P_Z = []
    for i in range(0, int(P_h/2)-1):
        f_i = int(row * i /P_h)
        for k in range(2):
            P_X_row = []
            P_Y_row = []
            P_Z_row = []
            for j in range(0, P_l -1):
                f_j = int(col * j / P_l)
                P_X_row.append(D_X[2*f_i+k][f_j])
                P_Y_row.append(D_Y[2*f_i+k][f_j])
                P_Z_row.append(D_Z[2*f_i+k][f_j])
            P_X_row.append(D_X[2*f_i+k][-1])
            P_Y_row.append(D_Y[2*f_i+k][-1])
            P_Z_row.append(D_Z[2*f_i+k][-1])
            P_X.append(P_X_row)
            P_Y.append(P_Y_row)
            P_Z.append(P_Z_row)

    for k in range(-2, 0):
        P_X_row = []
        P_Y_row = []
        P_Z_row = []
        for j in range(0, P_l - 1):
            f_j = int(col * j / P_l)
            P_X_row.append(D_X[k][f_j])
            P_Y_row.append(D_Y[k][f_j])
            P_Z_row.append(D_Z[k][f_j])
        P_X_row.append(D_X[k][-1])
        P_Y_row.append(D_Y[k][-1])
        P_Z_row.append(D_Z[k][-1])
        P_X.append(P_X_row)
        P_Y.append(P_Y_row)
        P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]
    return P


def LSPIA_FUNC_surface(file_name, D_h, D_l, P_h, P_l, miu):
    '''
       The LSPIA iterative method for blending surfaces.
       '''
    # D = load_surface_data('surface_data2')
    D_shadow_block = load_shadow_block_data(file_name)
    D_sample = sample_surface_data(D_shadow_block, D_h, D_l)
    D = [D_sample[0], D_sample[1], D_sample[2]]
    # D = [D_shadow_block[0], D_shadow_block[1], D_shadow_block[3]]

    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]

    row = len(D_X)
    col = len(D_X[0])

    p = 3  # degree
    q = 3

    '''
    Step 1. Calculate the parameters
    '''
    param_u = []
    tmp_param = np.zeros((1, row))
    for i in range(row):
        for j in range(col):
            tmp_param[0][i] = tmp_param[0][i] + D_Y[i][j]
        tmp_param[0][i] = tmp_param[0][i] / col
    tmp_param.sort()
    param_u = tmp_param.tolist()[0]

    param_v = []
    tmp_param = np.zeros((1, col))
    for j in range(col):
        for i in range(row):
            tmp_param[0][j] = tmp_param[0][j] + D_X[i][j]
        tmp_param[0][j] = tmp_param[0][j] / row
    tmp_param.sort()
    param_v = tmp_param.tolist()[0]
    # print(param_v)
    # print(param_u)

    '''
    Step 2. Calculate the knot vectors
    '''
    knot_uv = [[], []]
    knot_uv[0] = ps.LSPIA_knot_vector(param_u, p, P_h, row)
    knot_uv[1] = ps.LSPIA_knot_vector(param_v, q, P_l, col)
    # print(knot_uv[0])
    # print(knot_uv[1])

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
            f_j = int(col * j / P_l)
            P_X_row.append(D_X[f_i][f_j])
            P_Y_row.append(D_Y[f_i][f_j])
            P_Z_row.append(D_Z[f_i][f_j])
        P_X_row.append(D_X[f_i][-1])
        P_Y_row.append(D_Y[f_i][-1])
        P_Z_row.append(D_Z[f_i][-1])
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
    # c_u = np.zeros((1, P_h))
    for i in range(row):
        for j in range(P_h):
            Nik_u[i][j] = bf.BaseFunction(j, p + 1, param_u[i], knot_uv[0])
    #        c_u[0][j] = c_u[0][j] + Nik_u[i][j]
    # C = max(c_u[0].tolist())
    # miu_u = 2 / C

    Nik_v = np.zeros((col, P_l))
    # c_v = np.zeros((1, P_l))
    for i in range(col):
        for j in range(P_l):
            Nik_v[i][j] = bf.BaseFunction(j, q + 1, param_v[i], knot_uv[1])
    #         c_v[0][j] = c_v[0][j] + Nik_v[i][j]
    # C = max(c_v[0].tolist())
    # miu_v = 2 / C

    # miu = miu_u * miu_v
    # miu = 0.3
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
    piece_v = 100
    p_piece_u = np.linspace(param_u[0], param_u[-1], piece_u)
    p_piece_v = np.linspace(param_v[0], param_v[-1], piece_v)
    # p_piece_u = np.linspace(0, 1, piece_u)
    # p_piece_v = np.linspace(0, 1, piece_v)
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
    for i in range(int(row / 4)):
        for j in range(int(col / 4)):
            ax.scatter(D_X[4 * i][4 * j], D_Y[4 * i][4 * j], D_Z[4 * i][4 * j], color='r')
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


def LSPIA_FUNC_surface_for_sample_data(file_name, P_h, P_l, miu):
    '''
    The LSPIA iterative method for blending surfaces.
    Deal with the cross filed which is arranged as following:
    #    #    #   #    #
      *    *    *   *
    #    #    #   #    #
      *    *    *   *
    #    #    #   #    #
      *    *    *   *
    The even row has one more heliostat than odd row.
    Assumption:
        1. The number of row is even;
        2. The number of first row is even;
    Note:
         读取数据点后，对数据点进行采样，仅提供部分数据点用作LSPIA拟合；
    '''
    D_shadow_block = load_shadow_block_data(file_name)
    D = [D_shadow_block[0], D_shadow_block[1], D_shadow_block[3]]
    D_inter = cross_data_preprocess(D)

    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]

    D_inter_X = D_inter[0]
    D_inter_Y = D_inter[1]
    D_inter_Z = D_inter[2]

    row = len(D_inter_X)
    col = len(D_inter_X[0])
    col_even = len(D_X[0])
    col_odd = len(D_X[0]) - 1
    p = 3  # degree
    q = 3
    # P_h = int(row - 15)  # the number of control points
    # P_l = int(col_even - 40)

    '''
    Step 1. Calculate the parameters
    '''
    param_u = []
    tmp_param = np.zeros((1, row))
    for i in range(row):
        for j in range(col):
            tmp_param[0][i] = tmp_param[0][i] + D_inter_Y[i][j]
        tmp_param[0][i] = tmp_param[0][i] / col
    tmp_param.sort()
    param_u = tmp_param.tolist()[0]

    param_v = []
    tmp_param = np.zeros((1, col))
    for j in range(col):
        for i in range(row):
            tmp_param[0][j] = tmp_param[0][j] + D_inter_X[i][j]
        tmp_param[0][j] = tmp_param[0][j] / row
    tmp_param.sort()
    param_v = tmp_param.tolist()[0]
    # print(param_u)
    # print(param_v)

    '''
    Step 2. Calculate the knot vectors
    '''
    knot_uv = [[], []]
    knot_uv[0] = ps.LSPIA_knot_vector(param_u, p, P_h, row)
    knot_uv[1] = ps.LSPIA_knot_vector(param_v, q, P_l, col)
    # print(knot_uv[0])
    # print(knot_uv[1])

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
            f_j = int(col * j / P_l)
            P_X_row.append(D_inter_X[f_i][f_j])
            P_Y_row.append(D_inter_Y[f_i][f_j])
            P_Z_row.append(D_inter_Z[f_i][f_j])
        P_X_row.append(D_inter_X[f_i][-1])
        P_Y_row.append(D_inter_Y[f_i][-1])
        P_Z_row.append(D_inter_Z[f_i][-1])
        P_X.append(P_X_row)
        P_Y.append(P_Y_row)
        P_Z.append(P_Z_row)

    P_X_row = []
    P_Y_row = []
    P_Z_row = []
    for j in range(0, P_l - 1):
        f_j = int(col * j / P_l)
        P_X_row.append(D_inter_X[-1][f_j])
        P_Y_row.append(D_inter_Y[-1][f_j])
        P_Z_row.append(D_inter_Z[-1][f_j])
    P_X_row.append(D_inter_X[-1][-1])
    P_Y_row.append(D_inter_Y[-1][-1])
    P_Z_row.append(D_inter_Z[-1][-1])
    P_X.append(P_X_row)
    P_Y.append(P_Y_row)
    P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]

    '''
    Step 4. Calculate the collocation matrix of the NTP blending basis
    '''
    Nik_u = np.zeros((row, P_h))
    Nik_v_even = np.zeros((col_even, P_l))
    Nik_v_odd = np.zeros((col_odd, P_l))
    for i in range(row):
        for j in range(P_h):
            Nik_u[i][j] = bf.BaseFunction(j, p + 1, D_Y[row - 1 - i][0], knot_uv[0])
    for i in range(col_even):
        for j in range(P_l):
            Nik_v_even[i][j] = bf.BaseFunction(j, q + 1, D_X[0][i], knot_uv[1])
    for i in range(col_odd):
        for j in range(P_l):
            Nik_v_odd[i][j] = bf.BaseFunction(j, q + 1, D_X[1][i], knot_uv[1])
    Nik = [Nik_u, Nik_v_even, Nik_v_odd]
    # miu = 0.12

    '''
    Step 5. First iteration
    '''
    e = []
    ek = sfe.surface_fitting_error(D, P, Nik)
    e.append(ek)

    # return ek

    '''
    Step 6. Adjusting control points
    '''
    P = sfe.surface_adjusting_control_points(D, P, Nik, miu)
    # print(P)
    ek = sfe.surface_fitting_error(D, P, Nik)
    e.append(ek)

    cnt = 0
    while (abs(e[-1] - e[-2]) >= 1e-3):
        cnt = cnt + 1
        print('iteration ', cnt)
        P = sfe.surface_adjusting_control_points(D, P, Nik, miu)
        # print(P)
        ek = sfe.surface_fitting_error(D, P, Nik)
        e.append(ek)
    MSE = ek / (row * col - int(row / 2))
    print(MSE)
    error_matrix, error_list = sfe.point_fitting_error(D, P, Nik)

    with open('error_list_' + str(P_h) + 'x' + str(P_l) + '_' + str(miu) + '.txt', 'w') as file:
        for i in range(len(e) - 1):
            file.write(str(e[i]) + '\n')
        file.write(str(e[-1]))

    '''
    Step 7. Calculate data points on the b-spline curve
    '''
    piece_u = 60
    piece_v = 60
    p_piece_u = np.linspace(param_u[0], param_u[-1], piece_u)
    p_piece_v = np.linspace(param_v[0], param_v[-1], piece_v)

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

    # x_list = []
    # y_list = []
    # for i in range(row):
    #     x_list.extend(D_X[i])
    #     y_list.extend(D_Y[i])
    # error_list = np.array(error_list)
    # cm = plt.cm.get_cmap('seismic')
    # sc = plt.scatter(x_list, y_list, c=error_list, alpha=0.8, s=20, cmap=cm)
    # plt.colorbar(sc)
    # plt.show()

    p_piece_u_r = [p_piece_u[i] for i in range(piece_u - 1, -1, -1)]
    X, Y = np.meshgrid(p_piece_v, p_piece_u_r)
    Z = np.array(P_piece[2])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


fieldfile_path = 'field_data/cross_field/300x300/'
file_name = fieldfile_path + 'shadow_block_m1_d1_h8_min0.txt'

LSPIA_FUNC_surface(file_name, 100, 100, 50, 50, 0.3)
