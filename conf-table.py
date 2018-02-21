#!/usr/bin/env python3

import sys
from sklearn.metrics import confusion_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def read_labels(f):
    with open(f) as fp:
        lab = []
        for line in fp:
            lab.append(int(line.strip()))
    return lab

# def maximize_diag(m):
#     no_diag = dict()
#     for i in range(m.shape[0]):
#         for j in range(i+1, m.shape[0]):
#             print(i, j)
#             no_diag[(i,j)] = m[i,j] + m[j,i]
#     return no_diag


gold = read_labels(sys.argv[1])
pred = read_labels(sys.argv[2])

# gold = read_labels("data/es_test.labels")
# pred = read_labels("results/svm-spanish.output.txt")


labels = sorted(set(gold))
cm = confusion_matrix(gold, pred)

# labels = ["\\emoji{{es}}{{{}}}".format(x) for x in labels]
# fmt = "{:>3}" + "&{:>4}" * len(labels)
# print(fmt.format(" ", *labels))
# for i, row in enumerate(cm):
#     print(fmt.format(labels[i], *row))
# 

print("x,y,v")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        print ("{},{},{}".format(i, j, cm[i,j]))
    print()
