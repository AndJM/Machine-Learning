"""Closest pairs and clustering algorithms"""


import cluster
from random import randrange, uniform


def gen_random_clusters(num_clusters):
    """
    Return a list of clusters with centers corresponding to points of the unit
    square.
    """
    return [cluster.Cluster(set([]),
                            uniform(-1, 1),
                            uniform(-1, 1),
                            randrange(1, 101),
                            randrange(1, 101))
            for _ in range(num_clusters)]


def compute_distortion(cluster_list, data_table):
    return sum(clstr.cluster_error(data_table) for clstr in cluster_list)


def slow_closest_pair(cluster_list):
    """
    Compute the distance between a closest pair of clusters in cluster_list
    using an O(n^2) brute-force method.

    Return a tuple (min_dist, idx1, idx2) where cluster_list[idx1]
    and cluster_list[idx2] are closest with idx1 < idx2.
    """
    min_dist = (float('inf'), -1, -1)
    for idx1 in range(len(cluster_list) - 1):
        for idx2 in range(idx1 + 1, len(cluster_list)):
            sqrdist = cluster_list[idx1].sqrd_dist(cluster_list[idx2])
            min_dist = min(min_dist, (sqrdist**0.5, idx1, idx2))
    return min_dist


def fast_closest_pair(cluster_list):
    """
    Compute the distance between a closest pair of clusters in cluster_list
    using an O(nlog(n)) divide and conquer algorithm.
    """
    def fast_helper(cluster_list, horiz_order, vert_order):
        """
        Helper function to be called recursively.

        horiz_order and vert_order are lists of indices of clusters sorted by
        cluster centers; this accelerates the process of selecting clusters
        from cluster_list.

        Return a tuple (min_dist, idx1, idx2) where cluster_list[idx1]
        and cluster_list[idx2] are closest.
        """
        if len(horiz_order) < 4:
            clusters = [cluster_list[idx] for idx in horiz_order]
            min_dist, idx1, idx2 = slow_closest_pair(clusters)
            return (min_dist, horiz_order[idx1], horiz_order[idx2])

        m = int(len(horiz_order) / 2)
        mid_line = (cluster_list[horiz_order[m - 1]].horiz_center()
                    + cluster_list[horiz_order[m]].horiz_center()) / 2

        # separate points into a left and right partition retaining vertical
        # and horizontal order
        VL = []
        VR = []
        for idx in vert_order:
            if idx in horiz_order[:m]:
                VL.append(idx)
            else:
                VR.append(idx)

        # determine the smallest pairwise distance on the left and the right
        # side of the mid-line recursively
        left_pair = fast_helper(cluster_list, horiz_order[:m], VL)
        right_pair = fast_helper(cluster_list, horiz_order[m:], VR)
        min_dist = min(left_pair, right_pair)

        # consider the points on either side of the mid-line that are at most
        # min_dist apart
        remaining = []
        for idx in vert_order:
            if abs(cluster_list[idx].horiz_center() - mid_line) < min_dist[0]:
                remaining.append(idx)

        # iterate through the remaining points, checking the distance to at
        # most three lying above the current point
        k = len(remaining)
        for u in range(k - 1):
            for v in range(u + 1, min(u + 4, k)):
                sqrdist = cluster_list[remaining[u]].sqrd_dist(cluster_list[remaining[v]])
                min_dist = min(min_dist, (sqrdist**0.5, remaining[u], remaining[v]))

        return min_dist

    horiz_order = sorted(range(len(cluster_list)),
                         key=lambda x: cluster_list[x].horiz_center())
    vert_order = sorted(range(len(cluster_list)),
                        key=lambda x: cluster_list[x].vert_center())

    return fast_helper(cluster_list, horiz_order, vert_order)


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters. Mutates
    cluster_list.

    Input: List of clusters, number of clusters
    Output: List of clusters of length num_clusters
    """
    while len(cluster_list) > num_clusters:
        _, idx1, idx2 = fast_closest_pair(cluster_list)
        cluster_list[idx1].merge_clusters(cluster_list[idx2])
        del cluster_list[idx2]
    return cluster_list


def min_pair(cluster_obj, cluster_list):
    """
    Compute the distance between a cluster_obj and a cluster_list and
    return the index of the element in cluster_list closest to cluster_obj.
    """
    def dist(x): return (cluster_obj.sqrd_dist(cluster_list[x])**0.5)
    return min(range(len(cluster_list)), key=dist)


def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters.

    The initial points used as the centers are the clusters with largest
    population.

    Input: List of clusters, number of clusters, number of iterations
    Output: List of clusters of length num_clusters
    """
    srtd_clstrs = sorted(cluster_list,
                         key=lambda x: x.total_population(), reverse=True)
    centers = srtd_clstrs[:num_clusters]
    for _ in range(num_iterations):
        clstrs = [cluster.Cluster(set([]),
                                  cntr.horiz_center(),
                                  cntr.vert_center(),
                                  0,
                                  cntr.averaged_risk())
                  for cntr in centers]
        for c in cluster_list:
            min_idx = min_pair(c, centers)
            clstrs[min_idx].merge_clusters(c)
        centers = clstrs
    return clstrs
