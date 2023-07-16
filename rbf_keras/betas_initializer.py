from keras.initializers import Initializer
from scipy.spatial.distance import pdist
from tensorflow import constant, float32
from numpy import max, sqrt


class InitBetas(Initializer):
    """ Initializer for initialization of widths of RBF network
        using the vectors of centers.
    # Arguments
        C: matrix, with rows containing center vectors
    """

    def __init__(self, C):
        self.C = C

    def __call__(self, shape, dtype=None):
        dmax = max(pdist(self.C, 'euclidean'))
        P = self.C.shape[0]
        return constant(dmax/sqrt(P), shape=shape, dtype=float32)
