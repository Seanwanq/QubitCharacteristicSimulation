from scipy.linalg import expm
import numpy as np
from numpy import matrix
import sympy as sym


SigmaZ: matrix = matrix([[1, 0], [0, -1]])
SigmaY: matrix = matrix([[0, -1j], [1j, 0]])
SigmaX: matrix = matrix([[0, 1], [1, 0]])


def ComputeState(initialState: matrix, t: float, H: matrix):
    tempMatrix: matrix = -1j * H * t
    timeState: matrix = expm(tempMatrix) @ initialState
    return timeState


def ComputeStateProbability(state: matrix, chooser: int):
    if chooser == 0:
        probability: float = np.abs((state[0][0, 0]))**2
        return probability
    if chooser == 1:
        probability: float = np.abs((state[1][0, 0]))**2
        return probability
    else:
        probability: float = np.abs((state[2][0, 0]))**2
        return probability


def ComputeRamseyFinalStateWithoutNoise(initialState: matrix, H_d1: matrix, H_0: matrix, H_d2: matrix, A: float, τ: float, drive1Circle: float, drive2Circle: float):
    T: float = (2 * np.pi) / (2 * 0.5 * np.sqrt(A**2))
    t1: float = T * drive1Circle
    t2: float = T * drive2Circle
    firstState: matrix = ComputeState(initialState, t1, H_d1)
    secondState: matrix = ComputeState(firstState, τ, H_0)
    finalState: matrix = ComputeState(secondState, t2, H_d2)
    return finalState


def HamiltonianWithDrive2X2X(omegaQ: float, omegaD: float, phiD: float, A: float):
    delta: float = omegaD - omegaQ
    H: matrix = np.matrix([[0.5 * delta, 0.5 * A * np.exp(1j * phiD)],
                           [0.5 * A * np.exp(-1j * phiD), -0.5 * delta]])
    return H


def HamiltonianWithDrive2X2Y(omegaQ: float, omegaD: float, phiD: float, A: float):
    delta: float = omegaD - omegaQ
    H: matrix = np.matrix([[0.5 * delta, -0.5 * A * np.exp(1j * phiD)],
                           [-0.5 * A * np.exp(-1j * phiD), -0.5 * delta]])
    return H


def HamiltonianWithoutDrive2X2(omegaQ: float, omegaD: float):
    delta: float = omegaD - omegaQ
    H: matrix = np.matrix([[0.5 * delta, 0], [0, -0.5 * delta]])
    return H
