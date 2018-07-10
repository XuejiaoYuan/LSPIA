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
    # Nik_u = np.array(Nik[0])
    # Nik_v_max = np.array(Nik[1])
    # Nik_v_min = np.array(Nik[2])
    Nik_even_u = np.array(Nik[0][0])
    Nik_even_v = np.array(Nik[0][1])
    Nik_odd_u = np.array(Nik[1][0])
    Nik_odd_v = np.array(Nik[1][1])

    error = 0
    for dim in range(len(D)):
        D_dim_even = [D[dim][i] for i in range(0, len(D[dim]), 2)]
        D_dim_odd = [D[dim][i] for i in range(1, len(D[dim]), 2)]
        P_dim = np.array(P[dim])
        error = error + np.sum(np.square(D_dim_even - np.dot(np.dot(Nik_even_u, P_dim), np.transpose(Nik_even_v))))
        error = error + np.sum(np.square(D_dim_odd - np.dot(np.dot(Nik_odd_u, P_dim), np.transpose(Nik_odd_v))))

    # print(Nik_u.shape)
    # print(Nik_v.shape)
    # print(type(D[0]))
    # print(np.array(P[0]).shape)

    # for dim in range(len(D)):
    #     D_dim = np.array(D[dim])
    #     P_dim = np.array(P[dim])
    #     error = error + np.sum(np.square(D_dim - np.dot(np.dot(Nik_u, P_dim), np.transpose(Nik_v))))
    return error


def surface_adjusting_control_points(D, P, Nik, miu):
    '''
    Adjusting the surface control points with the adjusting vector.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: new control points
    '''
    Nik_even_u = np.array(Nik[0][0])
    Nik_even_v = np.array(Nik[0][1])
    Nik_odd_u = np.array(Nik[1][0])
    Nik_odd_v = np.array(Nik[1][1])

    for dim in range(len(D)):
        D_dim_even = [D[dim][i] for i in range(0, len(D[dim]), 2)]
        D_dim_odd = [D[dim][i] for i in range(1, len(D[dim]), 2)]
        P_dim = np.array(P[dim])
        delta_even_uv = D_dim_even - np.dot(np.dot(Nik_even_u, P_dim), np.transpose(Nik_even_v))
        delta_odd_uv = D_dim_odd - np.dot(np.dot(Nik_odd_u, P_dim), np.transpose(Nik_odd_v))
        delta = miu * (np.dot(np.transpose(Nik_even_u), np.dot(delta_even_uv, Nik_even_v))
                       + np.dot(np.transpose(Nik_odd_u), np.dot(delta_odd_uv, Nik_odd_v)))
        P[dim] = (P_dim + delta).tolist()
        # delta = miu * np.dot()

    # Nik_u = np.array(Nik[0])
    # Nik_v = np.array(Nik[1])
    # for dim in range(len(D)):
    #     D_dim = np.array(D[dim])
    #     P_dim = np.array(P[dim])
    #     delta_uv = D_dim - np.dot(np.dot(Nik_u, P_dim), np.transpose(Nik_v))
    #     delta = miu * np.dot(np.transpose(Nik_u), np.dot(delta_uv, Nik_v))
    #     P[dim] = (P_dim + delta).tolist()
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
        D.append(D_dim.tolist())
    return D
