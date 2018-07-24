"""
Implementation of many of `search.py` classes
"""
import copy
from search import *

class HWProblem(Problem):
    """
    An implementation of search.py Problem abstract class.
    """
    def __init__(self, task, no_del=False):
        super(HWProblem, self).__init__(task.initial_state)
        self.accept_task(task)
        self.no_del = False

    def accept_task(self, task):
        self.task = task
        self.initial = task.initial_state
        self.goal = task.goals

    # NOTE: must implement actions()
    def actions(self, state):
        """
        List of actions available, given the state
        """
        next_actions = self.task.get_successor_ops(state)
        return next_actions
        # for action in next_actions:
        #     yield action

    # NOTE: must implement results()
    def result(self, state, action):
        """
        Returns the resulting state as 
        action is taken on the given state
        """
        successor_list = self.task.get_successor_states(state, noDel = self.no_del)
        for act, result in successor_list:
            if act == action: return result

    def goal_test(self, state):
        # print "goal_state " + str(self.goal)
        # print "given state " + str(state)
        return self.goal <= state

    def value(self, state):
        return 1 # temporary implementation

    def get_node_path(self, node, terminate_at=None):
        '''
        Given a node, return a list of actions 
        executed to arrive at that node, from root
        '''
        path = []
        nodes = []
        current_node = node 
        while current_node.parent != terminate_at:
            path = [current_node.action] + path
            nodes = [current_node] + nodes
            current_node = current_node.parent
        return nodes, path

    def accept_solution_node(self, node):
        self.soln_node = node
        nodes, path = self.get_node_path(node)
        self.soln_nodelist = nodes
        self.soln_path = path
        self.soln_cost = 0

        # getting costs
        current_state = self.initial
        for n in nodes:
            self.soln_cost += self.path_cost(self.soln_cost, current_state, 
                                             n.action, n.state)
            current_state = n.state

#---
# Heuristics
#---

class HeuristicFF:
    def __init__(self, problem):
        self.problem = copy.deepcopy(problem)
        self.problem.no_del = True

    def get_RPG(self, node=None):
        '''
        Generate a relaxed planning graph
        '''
        if node is None: node = self.problem.initial
        t = 0
        F = [] # list of states
        A = [] # list of actions
        F.append(node.state)

        while not self.problem.goal <= F[-1]: 
            # if the goal still not in the last F
            A.append(list(self.problem.actions(F[-1])))
            f = copy.deepcopy(F[-1])
            for a in A[-1]:
                # loop through possible actions, 
                f = f.union(self.problem.result(F[-1], a))
            if f == F[-1]: 
                print "heur same state"
                break

            F.append(f)
            t += 1
        return F, A

    def get_first_level(self, F, goal=None):
        '''
        returns the index of the first layer where 
        the goal first appeared
        '''
        if goal is None: goal = self.problem.goal
        for i, f in enumerate(F):
            if goal in f: return i
        return -1

    def extract_RPG(self, F, A, goal=None):
        '''
        Extract the relaxed graph
        '''
        if goal is None: goal = self.problem.goal
        if not goal <= F[-1]: return float('inf')
        M = 0
        for g in goal:
            firstlevel = self.get_first_level(F, g)
            M = max(M, firstlevel)

        G = []
        for t in range(M+1):
            Gt = [g for g in goal
                  if self.get_first_level(F, g) == t]
            G.append(Gt)

        # print str(G) + " - " + str(M)

        selected_a = 0
        for t in range(M, 0, -1):
            for gt in G[t]:
                a = [act for act in self.problem.actions(F[t-1])
                         if gt in self.problem.result(F[t-1], act)
                            and self.get_first_level(A, goal=act) == t-1]
                if len(a) < 1: continue
                # print "action length " + str(len(a)) + " g " + str(gt)
                a_ = a[0]
                # print "action " + str(a_)
                # for a_ in a:
                for precon in a_.preconditions:
                    precon_t = self.get_first_level(F, precon)
                    G[precon_t - 1] += precon
                selected_a += 1
                # print "selected " + str(selected_a)
        return selected_a

    def __call__(self, node, goal=None):
        F, A = self.get_RPG(node=node)
        if F is None: return float('inf')
        selected_a = self.extract_RPG(F, A, goal=goal)
        return selected_a


# def get_hadd_function(problem, search_fn):

#     myproblem = copy.deepcopy(problem)
#     myproblem.no_del = True
    
#     def hadd(node):
#         myproblem.initial = node
#         soln_node = search_fn(myproblem)
#         if soln_node:
#             myproblem.accept_solution_node(soln_node)
#             return myproblem.soln_cost
#         else:
#             return float("inf")
#     return hadd #???



