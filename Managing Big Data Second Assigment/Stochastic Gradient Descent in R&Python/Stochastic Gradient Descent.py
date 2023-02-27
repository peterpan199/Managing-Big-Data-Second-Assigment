import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from random import seed
from random import randint


def stochasticGradientDescent(x, y, theta, alpha, m, numIters):
    # create and ampty array to save costs
    calculatedCosts = []
    ##choose one random observasion to use in sgd(use specialy number 4obs)
    i = random.randint(0, len(dependent) - 1)
    # Start the iteration
    for k in range(0, numIters):
        prediction= 0
        for j in range(10):
            prediction = prediction + (x[i, j]* thetas[j])
            ###calculate new theta
            newtheta = thetas[j] - alpha * (x[i, j]* (x[i, j]* thetas[j] - y[i]))
            theta[j] = newtheta
        ### calculate cost
        cost = ((prediction - y[i]) ** 2) / (2 * m)
        calculatedCosts.append(cost)
        print(f"===== Iteration ({k}) ======")
        print(theta)
        print(cost)
    return theta, calculatedCosts


Crime = pd.read_csv("communities.data", header=None, sep=",", engine='python')
# We find ViolentCrimesPerPop in column 127
print(Crime)
df1 = Crime.loc[:, [127]]
print(type(df1))
# Convert dataframe to matrix (ndarray)
dependent = df1.to_numpy()
print(type(dependent))
random_obs= randint(0, 1994)
# All the independent variables are in columns 17,26,27,31,32,37,76,90,95
df2 = Crime.loc[:, [17, 26, 27, 31, 32, 37, 76, 90, 95]]
print(type(df2))
# Convert dataframe to matrix (ndarray)
independentVars = df2.to_numpy()
print(type(independentVars))

### create and set the first column of independent variables as 1
onesColumn = np.ones((1994, 1))
independentVars = np.hstack((onesColumn, independentVars))
### set alpha, theta and number of iterations
alpha = 0.1
iniThetas = []
for i in range(0, independentVars.shape[1]):
    iniThetas.append( np.random.rand() )
thetas = np.array(iniThetas)
numIters = 10000

# The next step is to generate SGD.
estimatedThetas, costs = stochasticGradientDescent(independentVars, dependent, thetas, alpha, 1994, numIters)
print(">>> Estimated thetas")
print(estimatedThetas)

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(list(range(numIters)), costs, '-r')
plt.show()



























