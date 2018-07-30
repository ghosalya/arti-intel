# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MIT ca 2016

import pddl_parser
import search
import time
import sys
import pdb

def printOutputVerbose(tic, toc, path, cost, final_state, goal):
    print "\n******************************FINISHED TEST******************************"
    print "Goals: "
    for state in goal:
      print "\t" + str(state)
    print '\nRunning time: ', (toc-tic), 's'
    if path == None:
        print '\tNO PATH FOUND'
    else:
        print "\nNumber of Actions: ", len(path)
        print '\nCost:', cost
        print "\nPath: "
        for op in path:
          print "\t" + repr(op)
        print "\nFinal States:"
        for state in final_state:
          print "\t" + str(state)
    print "*************************************************************************\n"

def printOutput(tic, toc, path, cost):
    print (toc-tic), '\t', len(path), '\t', cost          

if __name__ == "__main__":
  args = sys.argv
  if len(args) != 3:                    # default task
      dirName = "prodigy-bw"
      fileName = "bw-simple"
  else:
      dirName = args[1]
      fileName = args[2]
  domain_file = dirName + '/domain.pddl'
  problem_file = dirName + '/' + fileName + '.pddl'

  # task is an instance of the Task class defined in strips.py
  task = pddl_parser.parse(domain_file, problem_file)

  # This should be commented out for larger tasks
  print task

  print "\n******************************START TEST******************************"
  

  # Define a sub-class of the Problem class, make an instance for the task and call the search
  # You should then set the variables:
  # final_state - the final state at the end of the plan
  # plan - a list of actions representing the plan
  # cost - the cost of the plan
  # Your code here

  from solved_search import HWProblem, HeuristicFF

  # Question 1
  print "\n++++++++++++++++++++++++++ QUESTION 1 ++++++++++++++++++++++++++++"
  tic = time.time()
  hw_prob = HWProblem(task)
  soln_node = search.breadth_first_search(hw_prob)
  hw_prob.accept_solution_node(soln_node)
  plan = hw_prob.soln_path
  cost = hw_prob.soln_cost
  final_state = hw_prob.soln_node.state

  toc = time.time()
  printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)
  
  

#   # Question 2
#   print "\n++++++++++++++++++++++++++ QUESTION 2 ++++++++++++++++++++++++++++"
#   tic = time.time()
#   hw_prob = HWProblem(task)
#   heur_fn = HeuristicFF(hw_prob)

#   soln_node = search.best_first_graph_search(hw_prob, heur_fn)
#   hw_prob.accept_solution_node(soln_node)
#   plan = hw_prob.soln_path
#   cost = hw_prob.soln_cost
#   final_state = hw_prob.soln_node.state

#   toc = time.time()
#   printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)


#   print "\n++++++++++++++++++++++++++ HEURISTIC COMPARISON ++++++++++++++++++++++++++++"

#   import winsound

#   dirName = "prodigy-bw"
#   problem_filenames = ["bw-simple",
#                        "bw-sussman",
#                     #    "bw-large-a",
#                     #    "bw-large-b",
#                     #    "bw-large-c",
#                     #    "bw-large-d"
#                        ]
#   tasks = [pddl_parser.parse(dirName + '/domain.pddl', dirName+'/'+probfile+'.pddl') 
#            for probfile in problem_filenames]

#   result = {}

#   print "\nComparison:"
#   for task in tasks:
#     hw_prob = HWProblem(task)
#     heur_fn = HeuristicFF(hw_prob)

#     start_node = search.Node(hw_prob.initial)
#     start_node.expand(hw_prob)
#     hff = heur_fn(start_node)
#     # print task.name + " " + str(hff)
    
#     hw_prob.no_del = False
#     soln_node = search.best_first_graph_search(hw_prob, heur_fn)
#     hw_prob.accept_solution_node(soln_node)
#     result[task.name] = (hff, len(hw_prob.soln_path))
#     print task.name + " hff:" + str(hff) + " plan: " + str(len(hw_prob.soln_path))
#     winsound.PlaySound("sound.wav", winsound.SND_FILENAME)
#     # winsound.MessageBeep()

  
