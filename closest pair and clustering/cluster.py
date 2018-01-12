"""
A helper class to implement data clustering methods.
"""


class Cluster:
    """Class for creating and merging clusters of counties"""
    def __init__(self, fips_codes, horiz_pos, vert_pos, population, risk):
        """Create a cluster based on county data."""
        self._fips_codes = fips_codes
        self._horiz_center = horiz_pos
        self._vert_center = vert_pos
        self._total_population = population
        self._averaged_risk = risk

    def __repr__(self):
        """String representation"""
        return "instance of {0}.Cluster\n{1!s}\n".format(__name__, self.__dict__)

    def fips_codes(self):
        """Get the cluster's set of FIPS codes"""
        return self._fips_codes

    def horiz_center(self):
        """Get the averged horizontal center of cluster"""
        return self._horiz_center

    def vert_center(self):
        """Get the averaged vertical center of the cluster"""
        return self._vert_center

    def total_population(self):
        """Get the total population for the cluster"""
        return self._total_population

    def averaged_risk(self):
        """Get the averaged risk for the cluster"""
        return self._averaged_risk

    def copy(self):
        """Return a copy of a cluster"""
        return Cluster(set(self._fips_codes),
                       self._horiz_center,
                       self._vert_center,
                       self._total_population,
                       self._averaged_risk,
                       )

    def sqrd_dist(self, cluster):
        """Compute the Euclidean distance between two clusters"""
        vert_dist = self._vert_center - cluster.vert_center()
        horiz_dist = self._horiz_center - cluster.horiz_center()
        return (vert_dist ** 2 + horiz_dist ** 2)

    def merge_clusters(self, cluster):
        """
        Merge two clusters. The populations of each are used to compute the new
        center and risk. Mutates self.
        """
        self._fips_codes.update(cluster.fips_codes())

        # compute weights for averaging
        pooled = self._total_population + cluster.total_population()
        prop1 = self._total_population / pooled
        prop2 = cluster.total_population() / pooled
        self._total_population = pooled

        # update center and risk using weights
        self._vert_center = prop1 * self._vert_center + \
            prop2 * cluster.vert_center()
        self._horiz_center = prop1 * self._horiz_center + \
            prop2 * cluster.horiz_center()
        self._averaged_risk = prop1 * self._averaged_risk + \
            prop2 * cluster.averaged_risk()

    def cluster_error(self, data_table):
        """
        data_table is the cancer data in which the data for this instance is
        located.

        Return the error as the sum of the squares of the distances from each
        county in this cluster to the center (weighted by the population).
        """
        # create hash table to accelerate error computation
        fips_to_row = {row[0]: idx for idx, row in enumerate(data_table)}

        # compute error as a weighted (squared) distance from each county to
        # the cluster center
        error = 0
        for county in self.fips_codes():
            row = data_table[fips_to_row[county]]
            cluster = Cluster(set([row[0]]), row[1], row[2], row[3], row[4])
            error += cluster.total_population() * self.sqrd_dist(cluster)
        return error
