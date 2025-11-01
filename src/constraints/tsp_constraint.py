class TSPConstraint:
    """
    Class to check constraints for a TSP solution.
    """

    def __init__(self, n_cities):
        self.n_cities = n_cities

    def is_valid_tour(self, tour):
        # Check if tour starts and ends at the same city
        if tour[0] != tour[-1]:
            return False
        # Check if all cities are visited exactly once (except start/end)
        visited = set(tour[:-1])
        if len(visited) != self.n_cities:
            return False
        # Check if all cities are in the valid range
        if not all(0 <= city < self.n_cities for city in visited):
            return False
        return True
