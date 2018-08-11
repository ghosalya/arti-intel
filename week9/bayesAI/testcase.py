from bayesian_network import BayesianNetwork, compute_p


print "=== TEST CASE 1 ==="
TESTCASE_1_singleProb = (("A", 0.99),)
TESTCASE_1_conditionalProb =(
    ("B", (
        ({"A":True}, 0.2),
        ({"A":False}, 0.8)
    )),
)



BN = BayesianNetwork(TESTCASE_1_singleProb, TESTCASE_1_conditionalProb)
compute_p(BN, {"B":True}, {"A":False})
compute_p(BN, {"B":True}, {})
compute_p(BN, {"B":True, "A":False}, {})




print "=== TEST CASE 1 ==="
TESTCASE_1_singleProb = (("A", 0.5),)
TESTCASE_1_conditionalProb =(
    ("B", (
        ({"A":True}, 0.2),
        ({"A":False}, 0.8)
    )),
)



BN = BayesianNetwork(TESTCASE_1_singleProb, TESTCASE_1_conditionalProb)
compute_p(BN, {"A":True}, {"B":True})


CONDITION = (
    ({"A":True}, 0.2),
    ({"A":False}, 0.6)
)

TESTCASE_2_singleProb = (("A", 0.5),)
TESTCASE_2_conditionalProb =(
    ("B", CONDITION),
    ("C", CONDITION),
    ("D", CONDITION),
)


n = BayesianNetwork(TESTCASE_2_singleProb, TESTCASE_2_conditionalProb)

print "\n=== TEST CASE 2 ==="
compute_p(n, {"A":False }, {"B":True, "C":True, "D":False})
compute_p(n, {"A":True }, {"B":True, "C":True, "D":False})

print "\n=== TEST CASE 2 ==="
compute_p(n, {"A":False }, {"B":False, "C":True, "D":False})
