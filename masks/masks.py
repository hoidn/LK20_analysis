import numpy as np
from dataccess import utils

#@utils.memoize(timeout = None)
def find_islands(arr, threshold, minsize = 75):
    """
    Remove 'islands' meeting the specified minimum size, and consisting of
    pixels with value above the given threshold, by setting them to 0.

    Returns the modified array and a list of ndarray of coordinate pairs for
    the affected pixels for each cluster/island:
        arr -> np.ndarray, clusters -> list of np.ndarray
    """
    arr = arr.copy()
    q = []
    def search(i, j):
        cluster = []
        q.append((i, j))
        visited[(i, j)] = True
        while q:
            if arr[(i + 1, j)] > threshold and not visited[(i + 1, j)]:
                q.append((i + 1, j))
                visited[(i + 1, j)] = True
            if arr[(i, j + 1)] > threshold and not visited[(i, j + 1)]:
                q.append((i, j + 1))
                visited[(i, j + 1)] = True
            if arr[(i - 1, j)] > threshold and not visited[(i - 1, j)]:
                q.append((i - 1, j))
                visited[(i - 1 , j)] = True
            if arr[(i, j - 1)] > threshold and not visited[(i, j -1 )]:
                q.append((i, j - 1))
                visited[(i , j - 1)] = True
            current = q.pop()
            i, j = current
            cluster.append(current)
        return cluster
    def zero_cluster(cluster):
        for coords in cluster:
            arr[coords] = 0
    visited = np.zeros(arr.shape, dtype = np.bool)
    clusters = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (not visited[i][j]) and (arr[i][j]) > 0:
                cluster = search(i, j)
                if len(cluster) < minsize:
                    zero_cluster(cluster)
                else:
                    clusters.append(cluster)
    return arr, map(np.array, clusters)

#@utils.memoize(timeout = None)
def bounding_view(array, cluster):
    def get_bounds(cluster):
        i, j = cluster.T
        return np.min(i), np.max(i), np.min(j), np.max(j)
    imin, imax, jmin, jmax = get_bounds(cluster)
    return array[imin:imax + 1, jmin:jmax + 1]

def clustermap(arr, clusters, func):
    def do_one(cluster):
        return bounding_view(arr, cluster).copy()
    return map(func, map(do_one, clusters))

def outlier_mask(arr, min = -10, max = 10):
    mask = np.ones(arr.shape, dtype = bool)
    mask[np.logical_or(arr < min, arr > max)] = False
    return mask

def grid_mask(arr, stride = 10):
    mask = np.ones(arr.shape, dtype = bool)
    mask[::stride, :] = False
    mask[:, ::stride] = False
    return mask

def write_masks():
    m1 = np.load('quad1_mask_calibman_10.12.npy')

    m2 = np.load('quad2_mask_calibman_10.12.npy')
    #m2 = np.load('mask3.npy')

    dark1 = np.load('dark1.npy').T
    new1 = outlier_mask(dark1) * m1
    #new1 = zero_islands(outlier_mask(dark1) * grid_mask(dark1) * m1, 0, 75)

    dark2 = np.load('dark2.npy').T
    new2 = outlier_mask(dark2) * m2
    #new2 = zero_islands(outlier_mask(dark2) * grid_mask(dark2) * m2, 0, 75)

    np.save('quad1_mask_10.11.npy', new1)

    np.save('quad2_mask_10.11.npy', new2)
