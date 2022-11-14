import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import datasets, metrics, tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# import warnings filter
from warnings import simplefilter
# ignore all FutureWarnings; may occur during predict() with k-nearest neighbours
simplefilter(action='ignore', category=FutureWarning)

# load data
iris = datasets.load_iris()
x = iris.data[:, 0] # sepal length
y = iris.data[:, 1] # sepal width
data = iris.data[:, :2]
classes = iris.target

# create scatter plot of data
x_min, x_max = x.min() - 0.5, x.max() + 0.5
y_min, y_max = y.min() - 0.5, y.max() + 0.5

plt.figure(figsize=(8,6))
plt.clf()

filled = [True, True, False]
colours = ['c', 'm', 'y']
markers = ['x', '+', 'o']

setosa = plt.scatter(x=x[np.nonzero(classes==0)], y=y[np.nonzero(classes==0)], c=colours[0], marker=markers[0])
versicolor = plt.scatter(x=x[np.nonzero(classes==1)], y=y[np.nonzero(classes==1)], c=colours[1], marker=markers[1])
virginica = plt.scatter(x=x[np.nonzero(classes==2)], y=y[np.nonzero(classes==2)], edgecolors=colours[2], marker=markers[2], facecolors='none')

plt.title("Sepal Length and Width of Irises")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend((setosa, versicolor, virginica),
           ("setosa", "versicolor", "virginica"),
           loc='lower left',
           ncol=1,
           fontsize=8)

plt.show()

################################################
# 10-Fold Cross-Validation, Nearest Neighbours #
################################################
print("Nearest Neighbours:")
kfold = KFold(n_splits=10, shuffle=True)
knc = KNeighborsClassifier(n_neighbors=10)
matrix = [[0 for i in np.unique(classes)] for i in np.unique(classes)]
for train, test in kfold.split(data): # indices of train and test data
    datatrain, datatest = data[train], data[test]
    classtrain, classtest = classes[train], classes[test]
    knc.fit(datatrain, classtrain)
    classpred = knc.predict(datatest) # FutureWarning may occur here
    confmat = metrics.confusion_matrix(classtest, classpred)
    matrix += confmat

# confusion matrix and accuracy stats
print("Confusion Matrix:")
print(matrix)
correct = 0
total = 0
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if i == j:
            correct += matrix[i][j]
        total += matrix[i][j]
print("Accuracy:", round(correct/total, 5))

print("\n" + "*"*30 + "\n")

############################################
# 10-Fold Cross-Validation, Decision Trees #
############################################

print("Decision Trees:")
kfold = KFold(n_splits=10, shuffle=True)
dtc = tree.DecisionTreeClassifier()
matrix = [[0 for i in np.unique(classes)] for i in np.unique(classes)]
for train, test in kfold.split(data): # indices of train and test data
    datatrain, datatest = data[train], data[test]
    classtrain, classtest = classes[train], classes[test]
    dtc.fit(datatrain, classtrain)
    classpred = dtc.predict(datatest)
    confmat = metrics.confusion_matrix(classtest, classpred)
    matrix += confmat

# confusion matrix and accuracy stats
print("Confusion Matrix:")
print(matrix)
correct = 0
total = 0
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if i == j:
            correct += matrix[i][j]
        total += matrix[i][j]
print("Accuracy:", round(correct/total, 5))

print("\n" + "*"*30 + "\n")

##################################################
# 10-Fold Cross-Validation, Gaussian Naive Bayes #
##################################################

print("Gaussian Naive Bayes:")
kfold = KFold(n_splits=10, shuffle=True)
gnb = GaussianNB()
matrix = [[0 for i in np.unique(classes)] for i in np.unique(classes)]
for train, test in kfold.split(data): # indices of train and test data
    datatrain, datatest = data[train], data[test]
    classtrain, classtest = classes[train], classes[test]
    gnb.fit(datatrain, classtrain)
    classpred = dtc.predict(datatest)
    confmat = metrics.confusion_matrix(classtest, classpred)
    matrix += confmat

# confusion matrix and accuracy stats
print("Confusion Matrix:")
print(matrix)
correct = 0
total = 0
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if i == j:
            correct += matrix[i][j]
        total += matrix[i][j]
print("Accuracy:", round(correct/total, 5))
