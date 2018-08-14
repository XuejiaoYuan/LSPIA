import numpy as np


def surface_fitting_error(D, P, Nik):
    '''
    Calculate the surface fitting error.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error
    '''
    error = point_fitting_error(D, P, Nik)
    return error


def point_fitting_error(D, P, Nik):
    '''
    Calculate the point fitting error.
    :param D: the data point
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error matrix
    '''
    error = 0

    # Nik_v: list
    row = len(D[0])

    # for dim in range(len(D)):
    for dim in range(1):
        D_dim_even = [D[dim][i] for i in range(0, len(D[dim]), 2)]
        D_dim_odd = [D[dim][i] for i in range(1, len(D[dim]), 2)]
        P_dim = np.array(P[dim])
        D_cal_even = np.dot(np.dot(Nik[0], P_dim), np.transpose(Nik[2]))
        D_cal_odd = np.dot(np.dot(Nik[1], P_dim), np.transpose(Nik[3]))
        delta_even_uv = D_dim_even - D_cal_even
        delta_odd_uv = D_dim_odd - D_cal_odd
        error = error + np.sum(np.square(delta_even_uv)) + np.sum(np.square(delta_odd_uv))


    # Nik_u = Nik[4]
    # for i in range(row):
    #     for dim in range(len(D)):
    #         Nik_u_row = np.array(Nik_u[i])
    #         P_dim = np.array(P[dim])
    #         if i % 2:
    #             Nik_v = Nik[3] # Nik[2]
    #         else:
    #             Nik_v = Nik[2] # Nik[1]
    #         D_cal = np.dot(np.dot(Nik_u_row, P_dim), np.transpose(Nik_v))
    #         delta = D[dim][i] - D_cal
    #         error = error + np.sum(np.square(delta))
    return error


def surface_adjusting_control_points(D, P, Nik, miu):
    '''
    Adjusting the surface control points with the adjusting vector.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: new control points
    '''
    Nik_u_even = Nik[0]
    Nik_u_odd = Nik[1]
    Nik_v_even = Nik[2]
    Nik_v_odd = Nik[3]

    for dim in range(len(D)):
        D_dim_even = [D[dim][i] for i in range(0, len(D[dim]), 2)]
        D_dim_odd = [D[dim][i] for i in range(1, len(D[dim]), 2)]
        P_dim = np.array(P[dim])
        delta_even_uv = D_dim_even - np.dot(np.dot(Nik_u_even, P_dim), np.transpose(Nik_v_even))
        delta_odd_uv = D_dim_odd - np.dot(np.dot(Nik_u_odd, P_dim), np.transpose(Nik_v_odd))
        delta = miu * (np.dot(np.transpose(Nik_u_even), np.dot(delta_even_uv, Nik_v_even))
                       + np.dot(np.transpose(Nik_u_odd), np.dot(delta_odd_uv, Nik_v_odd)))
        P_dim = P_dim + delta
        # if dim == 2:
        #     P_dim = np.where(P_dim < 0, 0, P_dim)
        P[dim] = P_dim.tolist()

    return P


def surface(param, P, Nik):
    '''
    Calculate the data points on the b-spline surface.
    :param param: the piece of param
    :param P: the control points
    :param Nik: the basis spline function
    :return: data points
    '''
    Nik_u = np.array(Nik[0])
    Nik_v = np.array(Nik[1])
    D = []
    for dim in range(len(P)):
        P_dim = np.array(P[dim])
        D_dim = np.dot(np.dot(Nik_u, P_dim), np.transpose(Nik_v))
        if dim == 2:
            D_dim = np.where(D_dim < 0, 0, D_dim)
        D.append(D_dim.tolist())
    return D