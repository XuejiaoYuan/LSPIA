import parameter_selection as ps
import BaseFunction as bf
import numpy as np
import fitting_error as fe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(filename):
    D_X = []
    D_Y = []
    with open(filename) as file:
        for line in file.readlines():
            line = line.strip()
            word = line.split(' ')
            D_X.append(float(word[0]))
            D_Y.append(float(word[1]))
    return [D_X, D_Y]


def LSPIA_curve():
    '''
    The LSPIA iterative method for blending curves.
    '''
    # D_X = [1, 1, 0, -0.5, 1.5,   3, 4, 4.2, 5, 5.5,  6, 5.1, 4.6]
    # D_Y = [0, 1, 2,    3,   4, 3.5, 3, 2.5, 2, 1.2,  1, 2.7,   4]
    # D = [D_X, D_Y]
    D = load_data('cur_data')
    D_X = D[0]
    D_Y = D[1]
    D_N = len(D_X)
    k = 3  # degree
    P_N = int(D_N / 2) - 50
    print(P_N)
    '''
    Step 1. Calculate parameters
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
    P = [P_X, P_Y]
    for i in range(1, P_N - 1):
        # f_i = int(D_N * i / P_N)
        # P_X.append(D_X[f_i])
        # P_Y.append(D_Y[f_i])
        P_X.append(0)
        P_Y.append(0)
    P_X.append(D_X[-1])
    P_Y.append(D_Y[-1])

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
    ek = fe.fitting_error(D, P, Nik)
    e.append(ek)

    '''
    Step 6. Adjusting control points
    '''
    P = fe.adjusting_control_points(D, P, Nik, miu)
    # print(P)
    ek = fe.fitting_error(D, P, Nik)
    e.append(ek)

    cnt = 0
    while (abs(e[-1] - e[-2]) >= 1e-7):
        cnt = cnt + 1
        print('iteration ', cnt)
        P = fe.adjusting_control_points(D, P, Nik, miu)
        # print(P)
        ek = fe.fitting_error(D, P, Nik)
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
    P_piece = fe.curve(p_piece, P, Nik_piece)

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



LSPIA_curve()

LSPIA_surface()