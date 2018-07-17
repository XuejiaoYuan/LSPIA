import numpy as np


def surface_fitting_error(D, P, Nik):
    '''
    Calculate the surface fitting error.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error
    '''
    error = 0
    error_matrix, error_list = point_fitting_error(D, P, Nik)
    error = np.sum(np.array(error_list))
    # Nik_u = Nik[0]
    # Nik_v_even = Nik[1]
    # Nik_v_odd = Nik[2]
    # row = len(D[0])
    #
    # for dim in range(len(D)):
    #     for i in range(row):
    #         Nik_u_row = np.array(Nik_u[i])
    #         P_dim = np.array(P[dim])
    #         if i % 2:
    #             error = error + np.sum(np.square(D[dim][i] - np.dot(np.dot(Nik_u_row, P_dim), np.transpose(Nik_v_odd))))
    #         else:
    #             error = error + np.sum(
    #                 np.square(D[dim][i] - np.dot(np.dot(Nik_u_row, P_dim), np.transpose(Nik_v_even))))
    return error


def point_fitting_error(D, P, Nik):
    '''
    Calculate the point fitting error.
    :param D: the data point
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error matrix
    '''
    error = []
    error_list = []
    Nik_u = Nik[0]
    Nik_v: list
    row = len(D[0])
    for i in range(row):
        # for j in range(col):
        for dim in range(len(D)):
            Nik_u_row = np.array(Nik_u[i])
            P_dim = np.array(P[dim])
            if i % 2:
                Nik_v = Nik[2]
            else:
                Nik_v = Nik[1]
            D_cal = np.dot(np.dot(Nik_u_row, P_dim), np.transpose(Nik_v))
            error_row = np.square(D[dim][i] - D_cal)
        error.append(error_row.tolist())
        error_list.extend(error_row.tolist())

    return error, error_list


def surface_adjusting_control_points(D, P, Nik, miu):
    '''
    Adjusting the surface control points with the adjusting vector.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: new control points
    '''
    row = len(D[0])
    col_even = len(D[0][0])
    col_odd = len(D[0][1])
    Nik_u = Nik[0]
    Nik_v_even = Nik[1]
    Nik_v_odd = Nik[2]

    Nik_u_even = np.array([Nik[0][i] for i in range(0, row, 2)])
    Nik_u_odd = np.array([Nik[0][i] for i in range(1, row, 2)])
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