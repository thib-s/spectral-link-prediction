import numpy as np


def path_counting(alpha: np.array([float]), x: np.array([float])): return np.dot(alpha, x)


def exponential(alpha: float, x: np.array([float])): return np.exp(alpha * x)


def exponential_odd(alpha: float, x: np.array([float])): return np.sinh(alpha * x)


def von_neumann(alpha: float, x: np.array([float])): return 1 / (1 - alpha * x)


def von_neumann_odd(alpha: float, x: np.array([float])): return alpha * x / (1 - alpha**2 * x**2)


def COM(x: np.array([float])): return 1 / np.dot(x, x[x > 0])


def COMR(alpha: float, x: np.array([float])): return 1 / (1 + alpha * x)


def HEAT(alpha: float, x: np.array([float])): return np.exp(-alpha * x)
