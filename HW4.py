import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
debug = True

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
def main():
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

    p1class1 = neighborRegression(p1nearest1)
    print(p1class1)

    p2class1 = neighborRegression(p2nearest1)
    print(p2class1)

    p1class3 = neighborRegression(p1nearest3)
    print(p1class3)

    p2class3 = neighborRegression(p2nearest3)
    print(p2class3)

    # plt.scatter(*zip(*[example.domain for example in trainingSet]), color = 'blue')
    # plt.scatter(*zip(p1, p2), color = 'red')
    #plotNearestNeighbors(trainingSet, (p1, p2))
    

if __name__ == "__main__":

    main()