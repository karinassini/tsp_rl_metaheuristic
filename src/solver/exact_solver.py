import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import math
import json


class TSPSolver:
    def __init__(self, dist, capitals):
        self.dist = dist
        self.capitals = capitals
        self.model = gp.Model()

    def build_model(self):
        # Create decision variables: is city 'i' adjacent to city 'j' on the tour?
        self.variables = self.model.addVars(
            self.dist.keys(), obj=self.dist, vtype=GRB.BINARY, name="x"
        )

        # Enforce symmetry: assign alias variables for reverse directions
        self.variables.update(
            {(j, i): self.variables[i, j] for i, j in self.variables.keys()}
        )

        # Add constraints in a separate method
        self.add_constraints()

    def add_constraints(self):
        # Each city must have exactly two incident edges
        self.model.addConstrs(
            self.variables.sum(city, "*") == 2 for city in self.capitals
        )

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(self, model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
            )
            # find the shortest cycle in the selected edge list
            tour = self.subtour(selected)
            if len(tour) < len(self.capitals):
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(
                    gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                    <= len(tour) - 1
                )

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(self, edges):
        unvisited = self.capitals[:]
        cycle = self.capitals[:]  # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle  # New shortest subtour
        return cycle

    def solve(self):
        self.build_model()
        self.model._vars = self.variables
        self.model.Params.lazyConstraints = 1
        self.model.optimize(self.subtourelim)
        print(self.model.display())
        return self.model, self.variables

    def get_solution(self):
        # Retrieve solution: extract variable values, build selected edges and tour
        vals = self.model.getAttr("x", self.variables)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = self.subtour(selected)
        assert len(tour) == len(self.capitals), "Incomplete tour"
        return tour, selected

    def save_solution(self, filename):
        tour, selected = self.get_solution()
        # Retrieve variable values from the model
        vals = self.model.getAttr("x", self.variables)
        # Compute the objective value (total distance) after the model has been optimized
        obj_val = self.model.ObjVal
        # Convert selected edges (a tuplelist) to a serializable list of tuples
        # Also convert the decision variable keys to strings for JSON serialization
        values = {f"{i},{j}": vals[i, j] for i, j in vals.keys()}
        sol = {
            "tour": tour,
            "selected_edges": list(selected),
            "objective_value": obj_val,
        }
        with open(filename, "w") as f:
            json.dump(sol, f)

        # Save the optimization model in LP format
        self.model.write(filename + ".lp")

    def load_solution(self, filename):
        with open(filename, "r") as f:
            sol = json.load(f)
        # Reconstruct the tuplelist for selected edges
        return sol["tour"], gp.tuplelist(sol["selected_edges"])
