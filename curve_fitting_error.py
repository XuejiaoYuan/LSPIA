import numpy as np


def curve_fitting_error(D, P, Nik):
    '''
    Calculate the curve fitting error.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: fitting error
    '''
    error = 0
    Nik = np.array(Nik)
    for dim in range(len(D)):
        D_dim = np.array(D[dim])
        P_dim = np.array(P[dim])
        error = error + np.sum(np.square(D_dim - np.dot(P_dim, np.transpose(Nik))))
    return error


def curve_adjusting_control_points(D, P, Nik, miu):
    '''
    Adjusting the curve control points with the adjusting vector.
    :param D: the data points
    :param P: the control points
    :param Nik: the basis spline function
    :return: new control points
    '''
    Nik = np.array(Nik)
    for dim in range(len(D)):
        D_dim = np.array(D[dim])
        P_dim = np.array(P[dim])
        delta = miu * np.dot(D_dim - np.dot(P_dim, np.transpose(Nik)), Nik)
        P[dim] = (P_dim + delta).tolist()
    return P


def curve(param, P, Nik):
    '''
    Calculate the data points on the b-spline curve.
    :param param: the piece of param
    :param P: the control points
    :param Nik: the basis spline function
    :return: data points
    '''
    Nik = np.array(Nik)
    D = []
    for dim in range(len(P)):
        P_dim = np.array(P[dim])
        D_dim = np.dot(P_dim, np.transpose(Nik))
        D.append(D_dim.tolist())
    return D