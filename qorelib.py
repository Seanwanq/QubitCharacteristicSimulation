from scipy.linalg import expm
import numpy as np
from numpy import polyfit, poly1d
from numpy import ndarray, matrix, complexfloating
import plotly.graph_objects as go
import random
from tqdm import tqdm
from scipy.optimize import leastsq


def ComputeTDState(initialState: matrix, t: float, H: matrix):
    tempMatrix: matrix = -1j * H * t
    timeState: matrix = expm(tempMatrix) @ initialState
    return timeState


def ComputeStateProbability(state: matrix, chooser: int):
    if chooser == 0:
        # probability: float =
