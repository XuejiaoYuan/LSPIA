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
    error_square = 0
    for dim in range(2, 3):
        D_dim_even = np.array([D[dim][i] for i in range(0, len(D[dim]), 2)])
        D_dim_odd = np.array([D[dim][i] for i in range(1, len(D[dim]), 2)])
        P_dim = np.array(P[dim])
        D_cal_even = np.dot(np.dot(Nik[0], P_dim), np.transpose(Nik[2]))
        D_cal_odd = np.dot(np.dot(Nik[1], P_dim), np.transpose(Nik[3]))
        delta_even_uv = D_dim_even - D_cal_even
        delta_odd_uv = D_dim_odd - D_cal_odd
        error_square = error_square + np.sum(np.square(delta_even_uv)) + np.sum(np.square(delta_odd_uv))
    return error_square


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

    for dim in range(2, 3):
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


def surface_fitting(D, P, Nik, miu, threashold):
    """
    Calculte the surface fitting's control points
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :param miu: parameter for control points adjusting
    :return:
    """
    D_even = [[], [], []]
    D_odd = [[], [], []]
    for dim in range(3):
        D_even[dim] = [D[dim][i] for i in range(0, len(D[dim]), 2)]
        D_odd[dim] = [D[dim][i] for i in range(1, len(D[dim]), 2)]

    D_even_odd = [D_even, D_odd]
    e_list = []

    error = surface_adjusting_control_points_new(D_even_odd, P, Nik, miu)
    e_list.append(error)

    error = surface_adjusting_control_points_new(D_even_odd, P, Nik, miu)
    e_list.append(error)
    while abs(e_list[-1] - e_list[-2]) > threashold:
        error = surface_adjusting_control_points_new(D_even_odd, P, Nik, miu)
        e_list.append(error)
        print('iteration: %d error: %f' % (len(e_list) + 1, error))
    return error


def surface_adjusting_control_points_new(D_even_odd, P, Nik, miu):
    """
    Calculate the point fitting error and adjust the control points.
    :param D:  the data points
    :param P: the control points
    :param Nik: the basis spline function
    :param miu: parameter for adjusting
    :param pre_e: last iteration's fitting error
    :return:
    """
    error = 0
    Nik_u_even = Nik[0]
    Nik_u_odd = Nik[1]
    Nik_v_even = Nik[2]
    Nik_v_odd = Nik[3]
    new_P = [[], [], []]
    for dim in range(2, 3):
        P_dim = np.array(P[dim])
        delta_even_uv = D_even_odd[0][dim] - np.dot(np.dot(Nik_u_even, P_dim), np.transpose(Nik_v_even))
        delta_odd_uv = D_even_odd[1][dim] - np.dot(np.dot(Nik_u_odd, P_dim), np.transpose(Nik_v_odd))
        delta = miu * (np.dot(np.transpose(Nik_u_even), np.dot(delta_even_uv, Nik_v_even))
                       + np.dot(np.transpose(Nik_u_odd), np.dot(delta_odd_uv, Nik_v_odd)))
        P_dim = P_dim + delta
        # new_P[dim] = P_dim.tolist()
        P[dim] = P_dim.tolist()
        error = error + np.sum(np.square(delta_even_uv)) + np.sum(np.square(delta_odd_uv))

    return error


def surface_fitting_all(D, P, param_uv, Nik, miu, threashold):
    """
    Calculte the surface fitting's control points
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :param miu: parameter for control points adjusting
    :return:
    """
    param_u = param_uv[0]
    param_v = param_uv[1]
    marker = np.zeros((len(param_u), len(param_v)))
    D_all = np.zeros((len(param_u), len(param_v)))
    map_u = {}
    map_v = {}
    for i in range(len(param_u)):
        map_u[param_u[i]] = i
    for i in range(len(param_v)):
        map_v[param_v[i]] = i
    for i in range(len(D[2])):
        for j in range(len(D[2][i])):
            f_i = map_u[D[1][i][j]]
            f_j = map_v[D[0][i][j]]
            D_all[f_i][f_j] = D[2][i][j]
            marker[f_i][f_j] = 1

    e_list = []

    error = surface_adjusting_control_points_new_all(D_all, marker, P, Nik, miu)
    e_list.append(error)

    error = surface_adjusting_control_points_new_all(D_all, marker, P, Nik, miu)
    e_list.append(error)
    while abs(e_list[-1] - e_list[-2]) > threashold:
        error = surface_adjusting_control_points_new_all(D_all, marker, P, Nik, miu)
        e_list.append(error)
        print('iteration: %d error: %f' % (len(e_list) + 1, error))
    return error


def surface_adjusting_control_points_new_all(D_all, marker, P, Nik, miu):
    """
    Calculate the point fitting error and adjust the control points.
    :param D:  the data points
    :param P: the control points
    :param Nik: the basis spline function
    :param miu: parameter for adjusting
    :param pre_e: last iteration's fitting error
    :return:
    """
    error = 0
    Nik_u = Nik[0]
    Nik_v = Nik[1]

    delta_uv = D_all - np.dot(np.dot(Nik_u, P[2]), np.transpose(Nik_v))
    delta_uv = delta_uv * marker
    delta = miu * np.dot(np.transpose(Nik_u), np.dot(delta_uv, Nik_v))
    P[2] = P[2] + delta
    error = error + np.sum(np.square(delta_uv))

    return error


def point_fitting_error_all(D, P, marker, Nik):
    '''
    Calculate the point fitting error.
    :param D: the data point
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error matrix
    '''

    D_cal = np.dot(np.dot(Nik[0], P[2]), np.transpose(Nik[1]))
    delta = D - D_cal
    delta = delta * marker
    error_square = np.sum(np.square(delta))
    return error_square
