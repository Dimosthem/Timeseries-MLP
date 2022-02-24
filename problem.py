import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
data = np.loadtxt('epilepticEEG.dat', unpack = True)

mvngAvg = np.zeros(len(data) - 500)
for i in range(500, len(data)):
    mvngAvg[i-500] = np.mean(data[i-500:i])



transformedData = np.zeros(len(data))
transformedData[0] = data[0]
for i in range(1, len(data)):
    transformedData[i] = data[i] - data[i-1]

mvngAvg = np.zeros(len(transformedData) - 500)
for i in range(500, len(transformedData)):
    mvngAvg[i-500] = np.mean(transformedData[i-500:i])

plt.plot(transformedData)
plt.show()

xTrain = pd.DataFrame({"X1": transformedData[0:3000], "X2": transformedData[1:3001], "X3": transformedData[2:3002]})
yTrain = transformedData[3:3003]
clf = MLPRegressor(hidden_layer_sizes=(500), max_iter = 10000)
clf = clf.fit(xTrain, yTrain)

#plt.plot (yTrain,"r")
norm_pred = clf.predict(xTrain)
pred = np.zeros(len(norm_pred))
pred[0] = norm_pred[0]
for i in range(1, len(pred)):
    pred[i] = norm_pred[i] + pred[i-1]
plt.plot(pred, "g")



plt.show()

trainingError = [(t - p) for (t, p) in zip(yTrain, clf.predict(xTrain))]
#βρίσκω το μέσο όρο του απόλυτου σφάλματος 
MAE = np.mean(np.abs(trainingError))

plt.plot(trainingError)
plt.ylabel('Training Error')
plt.xlabel('Sample')
plt.show()
print (MAE)

plt.hist(np.divide(trainingError, yTrain), density=True, bins=200)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Training Error %')
plt.show()


xTest = pd.DataFrame({"X1": data[3000:3397], "X2": data[3001:3398], "X3": data[3002:3399]})
yTest = data[3003:3400]
testingError = [(t - p) for (t, p) in zip(yTest, clf.predict(xTest))]

#plt.plot(testingError)
#plt.show()

plt.hist(np.divide(testingError, yTest), density=True, bins=200)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Testing Error %')
plt.show()

