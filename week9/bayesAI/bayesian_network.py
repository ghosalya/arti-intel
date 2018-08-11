from types import FloatType


## This class describe an event with probability value
## variables: conditions e.g. A = True, B = True / False, A = True & B = False, etc
## and the probability of this event happening
class ProbabilityEvent:
    def __init__(self, conditions, probability):
        self.conditions = conditions
        self.probability_value = probability
    
    ## Check if the input query for that event takes the event's legal value
    def check_event_value(self, input):
        for event, event_value in input.iteritems():
            if self.conditions[event] != event_value:
                return False
        return True
    
    ## For printing
    def __str__(self):
        return "%s: %.4f" % (self.conditions, self.probability_value)


class BayesianNetwork:

    ## The input of this network is a tuple
    ## ("event 1", probability_value) = prob event 1 is true, or
    ## conditional probability in this format: 
    ## ("event 1" , (
    ##              ({"condition 1":True}, probability value of event 1 is true given condition 1),
    ##              ({"condition 2 : False"}, probability value of event 1 is true given conditon 2),...
    ##              ))

    def __init__(self, single_event_probability, conditional_event_probability):

        ## initialize empty node
        self.nodes = [None]


        for name, probability in single_event_probability:
            print name, probability
            new_nodes = []
            true_case = dict()
            true_case[name] = True
            true_probability = probability
            false_case = dict()
            false_case[name] = False
            false_probability = 1 - probability

            new_nodes.append(ProbabilityEvent(true_case, true_probability))
            new_nodes.append(ProbabilityEvent(false_case, false_probability))

        self.nodes = new_nodes

        for n in self.nodes:
            print n

        for name, probabilities in conditional_event_probability:
            new_nodes = []
            for node in self.nodes:
                print node
                for given_name, prob_given in probabilities:
                    if node.check_event_value(given_name):
                        p = prob_given
                true_case = dict(node.conditions)
                true_case[name] = True
                true_probability = node.probability_value * p
                new_nodes.append(ProbabilityEvent(true_case, true_probability))

                false_case = dict(node.conditions)
                false_case[name] = False
                false_probability = node.probability_value * (1-p)
                new_nodes.append(ProbabilityEvent(false_case, false_probability))

            self.nodes = new_nodes
 
    
    ## Compute the probability of an event, given a series of events
    def P(self, event, given):
        prob_true = 0.0
        prob_false = 0.0
        for n in self.nodes:
            if n.check_event_value(given):
                if n.check_event_value(event):
                    prob_true += n.probability_value
                else:
                    prob_false += n.probability_value
        return prob_true / (prob_true + prob_false)


## helper method to print probability
true_false_symbol = {True:"", False:"-"}
def describe(item):
    return ','.join([true_false_symbol[value]+ name for name, value in item.iteritems()])


## method to compute probability, given the bayes network, the event, and a dictionary of given events
def compute_p(n, event, given={}):
    p = n.P(event, given)
    
    printout_string = describe(event)
    if given:
        printout_string += "|" + describe(given)
    
    print "P(%s) = %.4f" % (printout_string, p)

