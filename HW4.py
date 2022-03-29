import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
debug = False

class Instance:
    def __init__(self, domain, label) -> None:
        self.domain = domain
        self.label = label

def EuclideanDistance(p1, p2):
    dist = 0
    for n in range(len(p1)):
        dist += (p1[n]-p2[n])**2
    return dist**.5

def kNearestNeighbors(point, instances, k):
    distances = list()
    if debug: print(k,"nearest neighbors to", point)
    for instance in instances:
        dist = EuclideanDistance(point, instance.domain)
        distances.append((instance, dist))
        if debug: print("Point:", instance.domain, " Distance:", dist)
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    if debug: print("Nearest are:", ([n.domain for n in neighbors]))
    return neighbors

def neighborRegression(neighbors):
    h = 0
    for i in neighbors:
        h += i.label
    h /= len(neighbors)
    if debug: print("Label average:", h)
    return h

def plotNearestNeighbors(trainingSet, testSet):
    color = iter(cm.rainbow(np.linspace(0, 1, len(testSet))))
    for test in testSet:
        c = next(color)
        x_test = test[0]
        y_test = test[1]
        for train in trainingSet:
            x = train.domain[0]
            y = train.domain[1]
            plt.plot([x_test, x], [y_test, y], "o-", color=c, label = str(EuclideanDistance(test, train.domain)))
    plt.show()

def stringifyClassification(classification):
    if classification <= 0: return "-"
    else: return "+"
def Exercise1():
    e1 = Instance((-3, 5), 1)
    e2 = Instance((-4, 2), 1)
    e3 = Instance(( 2, 1), -1)
    e4 = Instance(( 4, 3), -1)
    trainingSet = [e1, e2, e3, e4]

    #Classify (-2, 0) and (-1, 5) using
    #a) 1-nearest neighbor
    #b) 3-nearest neighbor

    p1 = (-2, 0)
    p2 = (-1, 5)

    p1nearest1 = kNearestNeighbors(p1, trainingSet, 1)
    p2nearest1 = kNearestNeighbors(p2, trainingSet, 1)
    p1nearest3 = kNearestNeighbors(p1, trainingSet, 3)
    p2nearest3 = kNearestNeighbors(p2, trainingSet, 3)

    print("(-2, 0)")
    p1class1 = neighborRegression(p1nearest1)
    print("a)",stringifyClassification(p1class1))
    p2class1 = neighborRegression(p2nearest1)
    print("b)",stringifyClassification(p2class1))

    print("(-1, 5)")
    p1class3 = neighborRegression(p1nearest3)
    print("a)",stringifyClassification(p1class3))
    p2class3 = neighborRegression(p2nearest3)
    print("b)",stringifyClassification(p2class3))

    # plt.scatter(*zip(*[example.domain for example in trainingSet]), color = 'blue')
    # plt.scatter(*zip(p1, p2), color = 'red')
    #plotNearestNeighbors(trainingSet, (p1, p2))

def sgn(w, x):
    sum = 0
    for i in range(len(x)):
        sum += w[i] * x[i]
    if sum > 0: return 1
    else: return -1

def pUpdate(w, n, t, o, x):
    return w + n*(t-o)*x

def Exercise2():
    e1 = Instance((-3, 5), 1)
    e2 = Instance((-4, 2), 1)
    e3 = Instance(( 2, 1), -1)
    e4 = Instance(( 4, 3), -1)
    trainingSet = [e1, e2, e3, e4]
    #Starting weights w0=w1=w2=0
    #Learning rate n=0.1
    #3 iterations
    #What are the weights at the end of each iteration?
    iterations = 3
    weights = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    n = 0.1
    for i in range(iterations):
        for j in range(len(trainingSet)):
            w = weights[j]
            e = trainingSet[j]
            x = [1] + list(e.domain)
            t = e.label
            o = sgn(w, x)
            if o == t: continue
            else: 
                for k in range(len(w)):
                    w[k] = pUpdate(w[k], n, t, o, x[k])
        print(weights)
if __name__ == "__main__":
    Exercise2()