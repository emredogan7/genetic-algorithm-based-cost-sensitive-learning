import numpy as np
import random
from funcs import *
import matplotlib.pyplot as plt
from classify import classify
from oversampler import oversampler

# %%

# Preprocessing
trainData_ = np.loadtxt(fname="./../data/ann-train.data.txt")
testData = np.loadtxt(fname="./../data/ann-test.data.txt")

hnd = open('./../data/ann-thyroid.cost.txt', 'r')
costList = []
for line in hnd:
    costList.append(line.split()[1])
costList = list(map(float, costList))
costList.append(costList[18] + costList[19])

oversamplingFactor = 20
trainData = oversampler(trainData_, oversamplingFactor)
testData = oversampler(testData, oversamplingFactor)

X_train = trainData[:, :-1]
y_train = trainData[:, -1]

X_test = testData[:, :-1]
y_test = testData[:, -1]

print('\nPlease choose your classifier: \n')
print("""'knn' for k-nearest neighbor\n'nn' for neural network\n'dt' for decision tree """)
option = (input('Please enter your option for classifier:'))
##############################################################################

# %%
##############################################################################
# creating initial population

numOfInitMembers = 4

numOfInitMembers_ = numOfInitMembers

accuracyListPopulation = []
indicesListPopulation = []
costListPopulation = []
howManyFeatures = []
fitnessPopulation = []

while (numOfInitMembers_ != 0):

    genomeInd = np.random.randint(2, size=21).reshape((1, 21))
    while (np.sum(genomeInd) == 0):
        genomeInd = np.random.randint(2, size=21).reshape((1, 21))

    genomeInd = genomeInd.tolist()[0]
    indices = [i for i, x in enumerate(genomeInd) if x == 1]
    reducedX_train = X_train[:, indices]
    reducedX_test = X_test[:, indices]

    f1S, accTest = classify(option, reducedX_train, reducedX_test, y_train, y_test)
    accuracyListPopulation.append(accTest)
    indicesListPopulation.append(genomeInd)
    howManyFeatures.append(sum(genomeInd))
    costListPopulation.append(np.sum(np.multiply(genomeInd, costList)))
    fitnessPopulation.append(fitness(f1S, accTest, np.sum(np.multiply(genomeInd, costList))))
    numOfInitMembers_ -= 1

# %%
##############################################################################
# Genetic Algorithm
numberOfIterations = 5
numOfCouplesToCrossover = 1
numberOfIterations_ = numberOfIterations

ratioToMutate = 0.2
EachGenerationBest = []
EachGenerationAvg = []
while numberOfIterations_ != 0:
    print(numberOfIterations_)
    EachGenerationBest.append(max(fitnessPopulation))
    EachGenerationAvg.append(np.mean(fitnessPopulation))

    # crossover
    list_ = random.sample(indicesListPopulation, 2)
    parent1 = random.sample(indicesListPopulation, 2)[0]
    parent2 = random.sample(indicesListPopulation, 2)[1]

    offs1, offs2 = crossOver(parent1, parent2)
    while (np.sum(offs1) == 0 or np.sum(offs2) == 0):
        mutation(offs1)
        mutation(offs2)

    indices1 = [i for i, x in enumerate(offs1) if x == 1]
    reducedX_train1 = X_train[:, indices1]
    reducedX_test1 = X_test[:, indices1]

    f1S, accTest = classify(option, reducedX_train1, reducedX_test1, y_train, y_test)
    accuracyListPopulation.append(accTest)
    indicesListPopulation.append(offs1)
    howManyFeatures.append(sum(offs1))
    costListPopulation.append(np.sum(np.multiply(offs1, costList)))
    fitnessPopulation.append(fitness(f1S, accTest, np.sum(np.multiply(offs1, costList))))

    indices2 = [i for i, x in enumerate(offs2) if x == 1]
    reducedX_train2 = X_train[:, indices2]
    reducedX_test2 = X_test[:, indices2]

    f1S, accTest = classify(option, reducedX_train2, reducedX_test2, y_train, y_test)
    accuracyListPopulation.append(accTest)
    indicesListPopulation.append(offs2)
    howManyFeatures.append(sum(offs2))
    costListPopulation.append(np.sum(np.multiply(offs2, costList)))
    fitnessPopulation.append(fitness(f1S, accTest, np.sum(np.multiply(offs2, costList))))

    # mutation
    for x_ in range(int(ratioToMutate * numOfInitMembers)):
        pnt = np.random.randint(len(indicesListPopulation) - 1)

        mutation(indicesListPopulation[pnt])
        while (np.sum(indicesListPopulation[pnt]) == 0):
            mutation(indicesListPopulation[pnt])

        indicesMut = [i for i, x in enumerate(indicesListPopulation[pnt]) if x == 1]
        reducedX_trainMut = X_train[:, indicesMut]
        reducedX_testMut = X_test[:, indicesMut]

        f1S, accTest = classify(option, reducedX_trainMut, reducedX_testMut, y_train, y_test)
        accuracyListPopulation[pnt] = accTest

        howManyFeatures[pnt] = (sum(indicesListPopulation[pnt]))
        costListPopulation[pnt] = (np.sum(np.multiply(indicesListPopulation[pnt], costList)))
        fitnessPopulation[pnt] = fitness(f1S, accTest, np.sum(np.multiply(indicesListPopulation[pnt], costList)))

    minInd1 = fitnessPopulation.index(min(fitnessPopulation))

    del fitnessPopulation[minInd1], indicesListPopulation[minInd1]
    del costListPopulation[minInd1], howManyFeatures[minInd1]
    del accuracyListPopulation[minInd1]

    minInd2 = fitnessPopulation.index(min(fitnessPopulation))

    del fitnessPopulation[minInd2], indicesListPopulation[minInd2]
    del costListPopulation[minInd2], howManyFeatures[minInd2]
    del accuracyListPopulation[minInd2]

    numberOfIterations_ -= 1

print(fitnessPopulation)
qq_ = list(range(1, (len(EachGenerationAvg) + 1)))

plt.plot(qq_, EachGenerationAvg)
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")

plt.plot(qq_, EachGenerationBest)
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")