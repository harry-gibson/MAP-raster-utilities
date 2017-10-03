import itertools, math
import numpy as np


def getTiles(xSize, ySize, maxDim):
    ''' tile an input x and y dimension into tiles of specified maximum dimension

    Returns a list of tuples, one per tile. Each tuple has two sub 2-tuples reflecting the
    left-right and top-bottom boundaries of that tile.

    '''
    nchunksX = math.ceil(1.0 * xSize / maxDim)
    nchunksY = math.ceil(1.0 * ySize / maxDim)

    chunkedgesX = np.linspace(0, xSize, nchunksX + 1).astype(np.int32)
    chunkedgesY = np.linspace(0, ySize, nchunksY + 1).astype(np.int32)
    leftedges = chunkedgesX[:-1]
    rightedges = chunkedgesX[1:]
    topedges = chunkedgesY[:-1]
    bottomedges = chunkedgesY[1:]

    xoffsets = zip(leftedges, rightedges)
    yoffsets = zip(topedges, bottomedges)

    return list(itertools.product(xoffsets, yoffsets))