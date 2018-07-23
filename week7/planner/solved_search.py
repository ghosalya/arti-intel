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
        for action in next_actions:
            yield action

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

def get_hadd_function(problem, search_fn):
    myproblem = copy.deepcopy(problem)
    myproblem.no_del = True
    def hadd(node):
        myproblem.initial = node
        soln_node = search_fn(myproblem)
        if soln_node:
            myproblem.accept_solution_node(soln_node)
            return myproblem.soln_cost
        else:
            return float("inf")
    return hadd #???



