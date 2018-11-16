from scipy import  optimize as opt
import regression.functions as func

def fit(sigmaA: [], sigmaB: [], functionType: str):
    """
    perform the 1D regression between the eigenvalues of A and the eigenvalues of B
    :param sigmaA: array of the eig val of A
    :param sigmaB: idem for B
    :param functionType: the type of the function to use, see functions.py to see available functions
    :return: a function such that for all i: f(A[i]) =~= B[i]
    """
    pass