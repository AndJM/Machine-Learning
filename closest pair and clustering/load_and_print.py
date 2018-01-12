"""This module contains data loading and visualization functionality."""


# import urllib2
import matplotlib.pyplot as plt
from math import pi
from timeit import default_timer
from tools import *


PATH = "http://commondatastorage.googleapis.com/codeskulptor-assets/data_clustering/"
MAP = "USA_Counties.png"

# URLs to the tables of cancer risk data (the number of counties is indicated
# in the file name)
DATA_3108 = "unifiedCancerData_3108.csv"
DATA_896 = "unifiedCancerData_896.csv"
DATA_290 = "unifiedCancerData_290.csv"
DATA_111 = "unifiedCancerData_111.csv"
DATA_24 = "unifiedCancerData_24.csv"


# Define colors for clusters. Display a max of 16 clusters.
COLORS = ['Aqua', 'Yellow', 'Blue', 'Fuchsia', 'Black', 'Green',
          'Lime', 'Maroon', 'Navy', 'Olive', 'Orange', 'Purple',
          'Red', 'Brown', 'Teal']


def plot_clusters(data_table, cluster_list, draw_mode=0):
    """
    Create a plot of clusters of counties
    Draw mode 0: draw the counties colored by cluster
    Draw mode 1: use a single circle to represent counties in a cluster
    Draw mode 2: add cluster centers and lines from center to counties

    Code for visualizing clustering of county-based cancer risk data
    (modified from http://www.codeskulptor.org/#alg_clusters_matplotlib.py)
    """
    # build hash table to accelerate error computation
    fips_to_row = {row[0]: idx for idx, row in enumerate(data_table)}

    # Load map image
    # map_file = urllib2.urlopen(PATH + MAP)
    with open(MAP) as map_file:
        map_img = plt.imread(map_file)

    # Scale plot
    ypixels, xpixels, bands = map_img.shape
    DPI = 60.0  # adjust to resize plot
    xinch = xpixels / DPI
    yinch = ypixels / DPI
    plt.figure(figsize=(xinch, yinch))
    plt.imshow(map_img)

    def circle_area(p):
        """Helper function to compute circle area proportional to population"""
        return pi * p / (200.0 ** 2)

    if draw_mode == 0:
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_row[fips_code]]
                plt.scatter(x=[line[1]], y=[line[2]],
                            s=circle_area(line[3]), lw=1,
                            facecolors=cluster_color, edgecolors=cluster_color)

    elif draw_mode == 1:
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_center = (cluster.horiz_center(), cluster.vert_center())
            cluster_pop = cluster.total_population()
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            plt.scatter(x=[cluster_center[0]], y=[cluster_center[1]],
                        s=circle_area(cluster_pop), lw=1,
                        facecolors=cluster_color, edgecolors=cluster_color)

    elif draw_mode == 2:
        # draw county disks
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_row[fips_code]]
                plt.scatter(x=[line[1]], y=[line[2]],
                            s=circle_area(line[3]), lw=1,
                            facecolors=cluster_color, edgecolors=cluster_color,
                            zorder=1)

        # draw lines from cluster center to each county center
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            cluster_center = (cluster.horiz_center(), cluster.vert_center())
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_row[fips_code]]
                plt.plot([cluster_center[0], line[1]],
                         [cluster_center[1], line[2]],
                         cluster_color, lw=1, zorder=2)

        # draw circles at each cluster center
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_center = (cluster.horiz_center(), cluster.vert_center())
            cluster_pop = cluster.total_population()
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            plt.scatter(x=[cluster_center[0]], y=[cluster_center[1]],
                        s=circle_area(cluster_pop), lw=2,
                        facecolors="none", edgecolors="black",
                        zorder=3)
    plt.show()


def load_data_table(datafile):
    """
    Load a table of county-based cancer risk data from a csv format file
    and return a cluster list.
    """
    data = []
    # d = urllib2.urlopen(datafile)
    with open(datafile) as d:
        for line in d:
            r = line.split(',')
            data.append([r[0], float(r[1]), float(r[2]), int(r[3]), float(r[4])])
    print("Loaded", len(data), "data points.")
    return data


def make_cluster_list(data_table):
    return [cluster.Cluster(set([row[0]]),
                            row[1],
                            row[2],
                            row[3],
                            row[4])
            for row in data_table]


def make_plot1():
    """
    Running time comparison of slow_closest_pair and fast_closest_pair
    algorithms.
    """
    xvals = range(2, 201)
    yvals1 = []
    yvals2 = []

    import gc
    gc.disable()  # to remove the spikes in the plot

    for n in range(2, 201):
        cluster_list = gen_random_clusters(n)

        start_time = default_timer()
        slow_closest_pair(cluster_list)
        t = default_timer() - start_time
        yvals1.append(t)

        start_time = default_timer()
        fast_closest_pair(cluster_list)
        t = default_timer() - start_time
        yvals2.append(t)

    gc.enable()

    plt.plot(xvals, yvals1, '-r', label='slow_closest_pair')
    plt.plot(xvals, yvals2, '-b', label='fast_closest_pair')
    plt.title('Comparison of slow_closest_pair and fast_closest_pair \n'
              'running times on desktop Python')
    plt.xlabel('Number of randomly generated clusters')
    plt.ylabel('Time (sec)')
    plt.legend(loc='upper left')
    plt.show()


def make_plot2(draw_mode=0):
    """
    Use hierarchical clustering with 9 clusters on cancer data (111 counties)
    and superimpose the result on a provided map.
    """
    data_table = load_data_table(DATA_111)
    cluster_list = make_cluster_list(data_table)
    cluster_list = hierarchical_clustering(cluster_list, 9)
    print("Displaying", len(cluster_list), "hierarchical clusters.")
    plot_clusters(data_table, cluster_list, draw_mode)


def make_plot3(draw_mode=0):
    """
    Use kmeans clustering with 9 clusters and 5 iterations on cancer data
    (111 counties) and superimpose the result on a provided map.
    """
    data_table = load_data_table(DATA_111)
    cluster_list = make_cluster_list(data_table)
    cluster_list = kmeans_clustering(cluster_list, 9, 5)
    print("Displaying", len(cluster_list), "k-means clusters.")
    plot_clusters(data_table, cluster_list, draw_mode)


def make_plot4(draw_mode=0):
    """
    Use hierarchical clustering with 15 clusters on cancer data (3108 counties)
    and superimpose the result on a provided map.
    """
    data_table = load_data_table(DATA_3108)
    cluster_list = make_cluster_list(data_table)
    cluster_list = hierarchical_clustering(cluster_list, 15)
    print("Displaying", len(cluster_list), "hierarchical clusters.")
    plot_clusters(data_table, cluster_list, draw_mode)


def make_plot5(draw_mode=0):
    """
    Use kmeans clustering with 15 clusters and 5 iterations on cancer data
    (3108 counties) and superimpose the result on a provided map.
    """
    data_table = load_data_table(DATA_3108)
    cluster_list = make_cluster_list(data_table)
    cluster_list = kmeans_clustering(cluster_list, 15, 5)
    print("Displaying", len(cluster_list), "k-means clusters.")
    plot_clusters(data_table, cluster_list, draw_mode)


def make_plot6():
    """Compare the distortion of the clustering produced by both methods."""
    data_table = load_data_table(DATA_111)
    xvals = range(6, 21)
    yvals1 = []
    yvals2 = []
    for n in range(6, 21):
        cluster_list = make_cluster_list(data_table)

        cluster_list2 = kmeans_clustering(cluster_list, n, 5)
        print("Computing", len(cluster_list2), "k-means clusters.")
        yvals2.append(compute_distortion(cluster_list2, data_table))

        cluster_list1 = hierarchical_clustering(cluster_list, n)
        print("Computing", len(cluster_list1), "hierarchical clusters.")
        yvals1.append(compute_distortion(cluster_list1, data_table))

    plt.plot(xvals, yvals1, '-g', label='hierarchical_clustering')
    plt.plot(xvals, yvals2, '-b', label='k-means_clustering')
    plt.title('Comparison of distortion for clustering methods \n'
              'Data set: unifiedCancerData_111.csv')
    plt.xlabel('Number of output clusters')
    plt.ylabel('Distortion')
    plt.legend(loc='upper right')
    plt.show()


def make_plot7():
    """Compare the distortion of the clusterings produced by both methods."""
    data_table = load_data_table(DATA_290)
    xvals = range(6, 21)
    yvals1 = []
    yvals2 = []
    for n in range(6, 21):
        cluster_list = make_cluster_list(data_table)

        cluster_list2 = kmeans_clustering(cluster_list, n, 5)
        print("Computing", len(cluster_list2), "k-means clusters.")
        yvals2.append(compute_distortion(cluster_list2, data_table))

        cluster_list1 = hierarchical_clustering(cluster_list, n)
        print("Computing", len(cluster_list1), "hierarchical clusters.")
        yvals1.append(compute_distortion(cluster_list1, data_table))

    plt.plot(xvals, yvals1, '-g', label='hierarchical_clustering')
    plt.plot(xvals, yvals2, '-b', label='k-means_clustering')
    plt.title('Comparison of distortion for clustering methods \n'
              'Data set: unifiedCancerData_290.csv')
    plt.xlabel('Number of output clusters')
    plt.ylabel('Distortion')
    plt.legend(loc='upper right')
    plt.show()


def make_plot8():
    """Compare the distortion of the clusterings produced by both methods."""
    data_table = load_data_table(DATA_896)
    xvals = range(6, 21)
    yvals1 = []
    yvals2 = []
    for n in range(6, 21):
        cluster_list = make_cluster_list(data_table)

        cluster_list2 = kmeans_clustering(cluster_list, n, 5)
        print("Computing", len(cluster_list2), "k-means clusters.")
        yvals2.append(compute_distortion(cluster_list2, data_table))

        cluster_list1 = hierarchical_clustering(cluster_list, n)
        print("Computing", len(cluster_list1), "hierarchical clusters.")
        yvals1.append(compute_distortion(cluster_list1, data_table))

    plt.plot(xvals, yvals1, '-g', label='hierarchical_clustering')
    plt.plot(xvals, yvals2, '-b', label='k-means_clustering')
    plt.title('Comparison of distortion for clustering methods \n'
              'Data set: unifiedCancerData_896.csv')
    plt.xlabel('Number of output clusters')
    plt.ylabel('Distortion')
    plt.legend(loc='upper right')
    plt.show()


def compute_values():
    """
    Calculate the distortion associated with the 9 clusters produced by
    hierarchical and k-means clustering (with 5 iterations) on the 111 county
    data set.
    """
    data_table = load_data_table(DATA_111)
    cluster_list = make_cluster_list(data_table)
    km_cluster_list = kmeans_clustering(cluster_list, 9, 5)
    print("Distortion for kmeans on DATA_111: \n {}".format(
                            compute_distortion(km_cluster_list, data_table)))
    hc_cluster_list = hierarchical_clustering(cluster_list, 9)
    print("Distortion for hierarchical on DATA_111: \n {}".format(
                            compute_distortion(hc_cluster_list, data_table)))


if __name__ == '__main__':
    make_plot1()
    make_plot2()
    make_plot2(draw_mode=1)
    make_plot2(draw_mode=2)
    make_plot3()
    make_plot3(draw_mode=1)
    make_plot3(draw_mode=2)
    make_plot4()
    make_plot5()
    make_plot5(draw_mode=2)
    make_plot6()
    make_plot7()
    make_plot8()
    compute_values()
    exit()
