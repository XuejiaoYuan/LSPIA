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
from LSPIA import load_shadow_block_data, load_sd_bk_data
import time

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

plotly.tools.set_credentials_file(username='XuejiaoYuan', api_key='EILNngXAgk92tL3xeUy5')


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
    D_Z = D[3]
    row = len(D_X)
    col = len(D_X[0])

    P_X = []
    P_Y = []
    P_Z = []
    for i in range(0, int(P_h / 2) - 1):
        f_i = int(row * i / P_h)
        for k in range(2):
            P_X_row = []
            P_Y_row = []
            P_Z_row = []
            for j in range(0, P_l - 1):
                f_j = int(col * j / P_l)
                P_X_row.append(D_X[2 * f_i + k][f_j])
                P_Y_row.append(D_Y[2 * f_i + k][f_j])
                P_Z_row.append(D_Z[2 * f_i + k][f_j])
            P_X_row.append(D_X[2 * f_i + k][-1])
            P_Y_row.append(D_Y[2 * f_i + k][-1])
            P_Z_row.append(D_Z[2 * f_i + k][-1])
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


def sample_sd_bk_data(D, P_h, P_l):
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
    for i in range(0, int(P_h / 2) - 1):
        f_i = int(row * i / P_h)
        for k in range(2):
            P_X_row = []
            P_Y_row = []
            P_Z_row = []
            for j in range(0, P_l - 1):
                f_j = int(col * j / P_l)
                P_X_row.append(D_X[2 * f_i + k][f_j])
                P_Y_row.append(D_Y[2 * f_i + k][f_j])
                P_Z_row.append(D_Z[2 * f_i + k][f_j])
            P_X_row.append(D_X[2 * f_i + k][-1])
            P_Y_row.append(D_Y[2 * f_i + k][-1])
            P_Z_row.append(D_Z[2 * f_i + k][-1])
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
    D_shadow_block = load_sd_bk_data(file_name)
    D_sample = sample_sd_bk_data(D_shadow_block, D_h, D_l)
    D = [D_sample[0], D_sample[1], D_sample[2]]

    start_t = time.clock()

    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]

    row = len(D_X)
    col = len(D_X[0])
    row_even = int(col / 2) + col % 2
    row_odd = int(col / 2)

    p = 3  # degree
    q = 3

    '''
    Step 1. Calculate the parameters
    '''
    param_u = [y[0] for y in D_Y]
    param_u = sorted(param_u)
    param_v = np.linspace(D_X[0][0], D_X[0][-1], col)

    '''
    Step 2. Calculate the knot vectors
    '''
    knot_uv = [[], []]
    knot_uv[0] = ps.LSPIA_knot_vector(param_u, p, P_h, row)
    knot_uv[1] = ps.LSPIA_knot_vector(param_v, q, P_l, col)

    knot_u_front = []
    knot_u_end = []
    for i in range(int(len(knot_uv[0]) / 2)):
        knot_u_front.append(knot_uv[0][i])
    for i in range(int(len(knot_uv[0]) / 2), len(knot_uv[0])):
        knot_u_end.append(knot_uv[0][i])

    # fig1 = plt.figure()
    # plt.xticks(knot_u_front)
    # plt.ylim(-1, 1)
    # for i in range(int(len(param_u)/2)):
    #     plt.scatter(param_u[i], 0)
    # plt.show()
    #
    # fig2 = plt.figure()
    # plt.xticks(knot_u_end)
    # plt.ylim(-1, 1)
    # for i in range(int(len(param_u)/2), len(param_u)):
    #     plt.scatter(param_u[i], 0)
    # plt.show()
    #
    # knot_v_front = []
    # knot_v_end = []
    # for i in range(int(len(knot_uv[1])/2)):
    #     knot_v_front.append(knot_uv[1][i])
    # for i in range(int(len(knot_uv[1])/2), len(knot_uv[1])):
    #     knot_v_end.append(knot_uv[1][i])
    # fig3 = plt.show()
    # plt.xticks(knot_v_front)
    # plt.ylim(-1, 1)
    # for i in range(int(len(param_v)/2)):
    #     plt.scatter(param_v[i], 0)
    # plt.show()
    #
    # fig4 = plt.show()
    # plt.xticks(knot_v_end)
    # plt.ylim(-1, 1)
    # for i in range(int(len(param_v)/2), len(param_v)):
    #     plt.scatter(param_v[i], 0)
    # plt.show()

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
    P_X_row.append(D_X[-1][-1])
    P_Y_row.append(D_Y[-1][-1])
    P_Z_row.append(D_Z[-1][-1])
    P_X.append(P_X_row)
    P_Y.append(P_Y_row)
    P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]

    '''
    Step 4. Calculate the collocation matrix of the NTP blending basis
    '''
    Nik_u_even = np.zeros((row_even, P_h))
    Nik_u_odd = np.zeros((row_odd, P_h))
    Nik_v_even = np.zeros((col, P_l))
    Nik_v_odd = np.zeros((col, P_l))
    for i in range(row - 1, -1, -1):
        for j in range(P_h):
            if i % 2:
                Nik_u_odd[int(i / 2)][j] = bf.BaseFunction(j, p + 1, param_u[i], knot_uv[0])
            else:
                Nik_u_even[int(i / 2)][j] = bf.BaseFunction(j, p + 1, param_u[i], knot_uv[0])

    for i in range(col):
        for j in range(P_l):
            Nik_v_even[i][j] = bf.BaseFunction(j, q + 1, D_X[0][i], knot_uv[1])
    for i in range(col):
        for j in range(P_l):
            Nik_v_odd[i][j] = bf.BaseFunction(j, q + 1, D_X[1][i], knot_uv[1])
    Nik = [Nik_u_even, Nik_u_odd, Nik_v_even, Nik_v_odd]

    """
    Step 5. Calculate the fitting error and control points
    """
    error = sfe.surface_fitting(D, P, Nik, miu, 1e-4)
    MSE = np.sqrt(error / (D_h * D_l))
    print(MSE)

    # end_t = time.clock()
    # print(str(end_t))
    """
    Step 6. Calculate the error between data points and the fitting surface
    """
    row_data = len(D_shadow_block[0])
    col_data = len(D_shadow_block[0][0])

    row_data_even = int(row_data / 2) + row_data % 2
    row_data_odd = int(row_data / 2)
    Nik_piece_u_even = np.zeros((row_data_even, P_h))
    Nik_piece_u_odd = np.zeros((row_data_odd, P_h))
    Nik_piece_v_even = np.zeros((col_data, P_l))
    Nik_piece_v_odd = np.zeros((col_data - 1, P_l))
    for i in range(row_data - 1, -1, -1):
        for j in range(P_h):
            if i % 2:
                Nik_piece_u_odd[int(i / 2)][j] = bf.BaseFunction(j, p + 1,
                                                                 D_shadow_block[1][row_data - 1 - i][0], knot_uv[0])
            else:
                Nik_piece_u_even[int(i / 2)][j] = bf.BaseFunction(j, p + 1,
                                                                  D_shadow_block[1][row_data - 1 - i][0], knot_uv[0])
    for i in range(col_data):
        for j in range(P_l):
            Nik_piece_v_even[i][j] = bf.BaseFunction(j, q + 1, D_shadow_block[0][0][i], knot_uv[1])
    for i in range(col_data - 1):
        for j in range(P_l):
            Nik_piece_v_odd[i][j] = bf.BaseFunction(j, q + 1, D_shadow_block[0][1][i], knot_uv[1])
    Nik_piece = [Nik_piece_u_even, Nik_piece_u_odd, Nik_piece_v_even, Nik_piece_v_odd]
    error_square = sfe.point_fitting_error(
        [D_shadow_block[0], D_shadow_block[1], D_shadow_block[2]],
        P, Nik_piece
    )
    print(error_square)
    num = row_data * col_data - row_data_odd
    print(num)
    MSE = np.sqrt(error_square / num)
    print(MSE)

    '''
    Step 7. Calculate data points on the b-spline curve
    '''
    piece_u = 100
    piece_v = 100
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
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    p_piece_u_r = [p_piece_u[i] for i in range(piece_u - 1, -1, -1)]
    X, Y = np.meshgrid(p_piece_v, p_piece_u_r)
    Z = np.array(P_piece[2])
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    colorscale = [[0.0, 'rgb(20,29,67)'],
                  [0.1, 'rgb(28,76,96)'],
                  [0.2, 'rgb(16,125,121)'],
                  [0.3, 'rgb(92,166,133)'],
                  [0.4, 'rgb(182,202,175)'],
                  [0.5, 'rgb(253,245,243)'],
                  [0.6, 'rgb(230,183,162)'],
                  [0.7, 'rgb(211,118,105)'],
                  [0.8, 'rgb(174,63,95)'],
                  [0.9, 'rgb(116,25,93)'],
                  [1.0, 'rgb(51,13,53)']]
    trace1 = go.Surface(x=X, y=Y, z=Z, opacity=0.7, colorscale=colorscale)
    z_offset=(np.min(Z)-2)*np.zeros(Z.shape)
    proj_z = lambda x, y, z: z
    colorsurfz = proj_z(X,Y,Z)
    tracez = go.Surface(z=z_offset, x=X, y=Y,
                        colorscale=colorscale,
                        showlegend=False,
                        showscale=False,
                        surfacecolor=colorsurfz)
    layout = go.Layout(
        # scene = dict(
        # xaxis = dict(
        #      backgroundcolor="rgb(200, 200, 230)",
        #      gridcolor="rgb(255, 255, 255)",
        #      showbackground=True,
        #      zerolinecolor="rgb(255, 255, 255)",),
        # yaxis = dict(
        #     backgroundcolor="rgb(230, 200,230)",
        #     gridcolor="rgb(255, 255, 255)",
        #     showbackground=True,
        #     zerolinecolor="rgb(255, 255, 255)"),
        # zaxis = dict(
        #     backgroundcolor="rgb(230, 230,200)",
        #     gridcolor="rgb(255, 255, 255)",
        #     showbackground=True,
        #     zerolinecolor="rgb(255, 255, 255)",),),
        width=700,
        margin=dict(
            r=10, l=10,
            b=10, t=10))
    fig = go.Figure(data=[trace1, tracez], layout=layout)
    py.plot(fig)


def sample_surface_data_all(D, P_h, P_l):
    '''
    Sample the surface data
    :param D: the surface data
    :param P_h: the number of sample rows
    :param P_l: the number of sample cols
    :return: sample data
    '''
    D_X = D[0]
    D_Y = D[1]
    D_Z = D[3]
    row = len(D_X)

    P_X = []
    P_Y = []
    P_Z = []
    for i in range(0, P_h):
        f_i = int(row * i / P_h)
        P_X_row = []
        P_Y_row = []
        P_Z_row = []
        for j in range(0, P_l):
            f_j = int(len(D_X[f_i]) * j / P_l)
            P_X_row.append(D_X[f_i][f_j])
            P_Y_row.append(D_Y[f_i][f_j])
            P_Z_row.append(D_Z[f_i][f_j])
        P_X.append(P_X_row)
        P_Y.append(P_Y_row)
        P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]
    return P


def LSPIA_FUNC_surface_all(file_name, D_h, D_l, P_h, P_l, miu):
    '''
       The LSPIA iterative method for blending surfaces.
       '''
    D_shadow_block = load_shadow_block_data(file_name)
    D_sample = sample_surface_data_all(D_shadow_block, D_h, D_l)
    D = [D_sample[0], D_sample[1], D_sample[2]]

    start_t = time.clock()

    D_X = D[0]
    D_Y = D[1]
    D_Z = D[2]

    row = len(D_X)
    col = len(D_X[0])
    # row_even = int(col / 2) + col % 2
    # row_odd = int(col / 2)

    p = 3  # degree
    q = 3

    '''
    Step 1. Calculate the parameters
    '''
    param_u = [y[0] for y in D_Y]
    param_u = sorted(param_u)
    # param_v = np.linspace(D_X[0][0], D_X[0][-1], col)
    param_v = []
    for i in range(2):
        for j in range(len(D_X[i])):
            param_v.append(D_X[i][j])
    param_v = sorted(param_v)

    '''
    Step 2. Calculate the knot vectors
    '''
    knot_uv = [[], []]
    knot_uv[0] = ps.LSPIA_knot_vector(param_u, p, P_h, len(param_u))
    knot_uv[1] = ps.LSPIA_knot_vector(param_v, q, P_l, len(param_v))

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
    P_X_row.append(D_X[-1][-1])
    P_Y_row.append(D_Y[-1][-1])
    P_Z_row.append(D_Z[-1][-1])
    P_X.append(P_X_row)
    P_Y.append(P_Y_row)
    P_Z.append(P_Z_row)

    P = [P_X, P_Y, P_Z]

    '''
    Step 4. Calculate the collocation matrix of the NTP blending basis
    '''
    Nik_u = np.zeros((row, P_h))
    Nik_v = np.zeros((len(param_v), P_l))
    for i in range(row - 1, -1, -1):
        for j in range(P_h):
            Nik_u[i][j] = bf.BaseFunction(j, p + 1, param_u[i], knot_uv[0])

    for i in range(len(param_v)):
        for j in range(P_l):
            Nik_v[i][j] = bf.BaseFunction(j, q + 1, param_v[i], knot_uv[1])

    Nik = [Nik_u, Nik_v]

    """
    Step 5. Calculate the fitting error and control points
    """
    error = sfe.surface_fitting_all(D, P, [param_u, param_v], Nik, miu, 1e-4)
    MSE = np.sqrt(error / (D_h * D_l))
    print(MSE)

    # end_t = time.clock()
    # print(str(end_t))
    """
    Step 6. Calculate the error between data points and the fitting surface
    """
    data_row = len(D_shadow_block[0])
    data_col = len(D_shadow_block[0][0]) + len(D_shadow_block[0][1])

    param_piece_u = [y[0] for y in D_shadow_block[1]]
    param_piece_u = sorted(param_piece_u)
    param_piece_v = []
    for i in range(2):
        for j in range(len(D_shadow_block[0][i])):
            param_piece_v.append(D_shadow_block[0][i][j])
    param_piece_v = sorted(param_piece_v)
    map_piece_u = {}
    for i in range(len(param_piece_u)):
        map_piece_u[param_piece_u[i]] = i
    map_piece_v = {}
    for i in range(len(param_piece_v)):
        map_piece_v[param_piece_v[i]] = i
    D_total = np.zeros((data_row, data_col))
    marker = np.zeros((data_row, data_col))
    for i in range(len(D_shadow_block[2])):
        for j in range(len(D_shadow_block[2][i])):
            f_i = map_piece_u[D_shadow_block[1][i][j]]
            f_j = map_piece_v[D_shadow_block[0][i][j]]
            D_total[f_i][f_j] = D_shadow_block[3][i][j]
            marker[f_i][f_j] = 1

    Nik_piece_u = np.zeros((data_row, P_h))
    Nik_piece_v = np.zeros((data_col, P_l))

    for i in range(len(param_piece_u)):
        for j in range(P_h):
            Nik_piece_u[i][j] = bf.BaseFunction(j, p + 1, param_piece_u[i], knot_uv[0])
    for i in range(len(param_piece_v)):
        for j in range(P_l):
            Nik_piece_v[i][j] = bf.BaseFunction(j, q + 1, param_piece_v[i], knot_uv[1])

    error_square = sfe.point_fitting_error_all(
        D_total, P,
        marker, [Nik_piece_u, Nik_piece_v]
    )
    print(error_square)
    num = 89850
    print(num)
    MSE = np.sqrt(error_square / num)
    print(MSE)

    '''
    Step 7. Calculate data points on the b-spline curve
    '''
    piece_u = 100
    piece_v = 100
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
    p_piece_u_r = [p_piece_u[i] for i in range(piece_u)]
    X, Y = np.meshgrid(p_piece_v, p_piece_u_r)
    Z = np.array(P_piece[2])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


fieldfile_path = 'field_data/cross_field/300x300/'
for i in range(8, 18):
    file_name = fieldfile_path + 'clipper_m1_d25_h' + str(i) + '_min0.txt'
    LSPIA_FUNC_surface(file_name, 100, 100, 76, 76, 0.9)

# LSPIA_FUNC_surface_all(file_name, 100, 100, 76, 76, 0.9)
