from scipy import optimize as opt
import regression.functions as funcs
import numpy as np


def fit(sigmaA: np.array, sigmaB: np.array, functionType, verbose=False):
    """
    perform the 1D regression between the eigenvalues of A and the eigenvalues of B
    :param sigmaA: array of the eig val of A
    :param sigmaB: idem for B
    :param functionType: the type of the function to use, see functions.py to see available functions
    :return: the fitted function F such that for all i: f(A[i]) =~= B[i]
    """
    sigmaA.sort()
    sigmaB.sort()
    loss_func = lambda alpha: function_loss(functionType, sigmaB, sigmaA, alpha)
    res = opt.minimize(fun=loss_func, x0=np.array([0.]), bounds=np.array([(0., 1.)]))
    return lambda sigmaA: functionType(res.x[0], sigmaA)


def function_loss(function, sigmaB, sigmaA, alpha):
    """
    compute the square difference between function(sigmaB) and sigmaA.
    This the function that we will optimize
    :param function: the spectral function we want to fit
    :param sigmaB: spectrum of B
    :param sigmaA: spectrum of A
    :param alpha: parameter of the function
    :return: the square loss
    """
    return np.mean(np.square(function(alpha, sigmaA) - sigmaB))


if __name__ == '__main__':
    sigmaA = np.array([1., 2., 6., 4., 5., 5.5, 7.])
    sigmaB = funcs.exponential(0.5, sigmaA)
    F = fit(sigmaA, sigmaB, funcs.exponential)
    print(F(sigmaA))
    print(sigmaB)
