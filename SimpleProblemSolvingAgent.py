import sys
from collections import deque
import search as search
from utils import *

## Start of Search.py supplied code
class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal
    
    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        value = self.h(state)
        return value

# ______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with 'lowest' value,
    stopping when no neighbor is better.
    """
    # Modifications to supplied code
    # Returning a path vs. node and checking plateau
    current = Node(problem.initial)
    path = [current]
    visited = set()
    
    while not problem.goal_test(current.state):
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmin_random_tie(neighbors, key=lambda node: problem.value(node.state))
       # If the chosen neighbor is closer to the goal than the current node.
        if problem.value(neighbor.state) <= problem.value(current.state):
            # Add the current node to the visited set and move to the chosen neighbor.
            current = neighbor
            path.append(current)
        # If the chosen neighbor is not closer than the current node to the goal and we are not at the goal state.
        if problem.value(neighbor.state) > problem.value(current.state) and not problem.goal_test(current.state):
            # Add the current node to the visited set and break out of the loop.
            visited.add(current.state)
            path.append("Plateau")
            break
    return current

def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule(), max_attempts=5):
    """Simulated annealing with reruns if the goal is not reached."""
    best_solution = None
    """Loop through attempt number."""
    for attempt in range(max_attempts):
        current = Node(problem.initial)
        path = [current] 
       # store visited to ensure no repeated states
        visited = set([tuple(current.state)]) 
        for t in range(sys.maxsize): 
            T = schedule(t) 
            if T == 0:
                break
            # If the current node is the goal state, return the current node.
            if problem.goal_test(current.state):
                return current  
            
            # check neighbors
            neighbors = [
                neighbor for neighbor in current.expand(problem)
                # ensure no repeated states
                if tuple(neighbor.state) not in visited
            ]
    
            # If no neighbors, continue
            if not neighbors:
                continue 
            
            # Choose a random neighbor
            next_choice = random.choice(neighbors)
            
            # Calculate the change in value (delta_e)
            delta_e = problem.value(current.state) - problem.value(next_choice.state)
            
            # If the new path is better or accepted probabilistically
            if delta_e > 0 or np.random.rand() < np.exp(delta_e / T):
                current = next_choice
                visited.add(tuple(current.state))  # Mark the path as visited
                path.append(current)
        
        # If no solution was found in this attempt, keep track of the best solution found
        if best_solution is None or problem.value(current.state) < problem.value(best_solution.state):
            best_solution = current

    # Return the best found state after all attempts
    return best_solution

# ______________________________________________________________________________

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf
# -----------------------------------------------------------------------------
# End of Search.py supplied code

def printResult(res):
    node_states = [node.state if node.state != "Plateau" else "Local Min Reached" for node in res.path()]
    print(" → ".join(node_states))

    print("Cost " + str(res.path_cost))


def runSearchAlgorithms(problem):
    ## Greedy Best-First Search
    print("Greedy Best-First Search")

    printResult(best_first_graph_search(problem, problem.h, display=True))
    print("\n")

    ## A* Search
    print("A* Search")
    printResult(astar_search(problem))
    print("\n")

    ## Hill Climbing Search
    print("Hill Climbing Search")
    printResult(hill_climbing(problem))
    print("\n")

    ## Simulated Annealing Search
    print("Simulated Annealing Search")
    printResult(simulated_annealing(problem))
    print("\n")