import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

file = pandas.read_csv("IMDB Dataset.csv", error_bad_lines=False)

data = file["review"].values[:200]
labels = file["sentiment"].values[:200]

labels = np.where(labels == 'negative', 0, labels)
labels = np.where(labels == 'positive', 1, labels)

labels = labels.astype(float)

vectorizer = CountVectorizer()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)
x_train = x_train.toarray()
x_test = x_test.toarray()


def naive_bayes(xtrain, ytrain, xtest, ytest):
    from sklearn.naive_bayes import BernoulliNB

    gnb = BernoulliNB()
    model = gnb.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    percision = metrics.precision_score(ytest, y_pred, average="macro")
    recall = metrics.recall_score(ytest, y_pred, average="macro")
    f_massure = (2*percision*recall)/(percision+recall)
    accuracy = (percision*100,recall, f_massure, 'naive bayes')

    return accuracy


def Decision_Tree(xtrain, ytrain, xtest, ytest):
    from sklearn import tree
    d_tree = tree.DecisionTreeClassifier()
    model = d_tree.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    percision = metrics.precision_score(ytest, y_pred, average="macro")
    recall = metrics.recall_score(ytest, y_pred, average="macro")
    f_massure = (2 * percision * recall) / (percision + recall)
    accuracy = (percision*100,recall, f_massure,  'Decision Tree classifier')
    return accuracy


def rule_based_classification(xtrain, ytrain, xtest, ytest):
    from sklearn.dummy import DummyClassifier
    rbc = DummyClassifier(strategy="most_frequent")
    model = rbc.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    percision = metrics.precision_score(ytest, y_pred, average="macro")
    recall = metrics.recall_score(ytest, y_pred, average="macro")
    f_massure = (2 * percision * recall) / (percision + recall)
    accuracy = (percision*100,recall, f_massure, 'rule based classification')

    return accuracy


NB = naive_bayes(x_train, y_train, x_test, y_test)
DT = Decision_Tree(x_train, y_train, x_test, y_test)
Rbc = rule_based_classification(x_train, y_train, x_test, y_test)

print(NB, DT, Rbc)

width = 0.35
p = plt.bar(np.arange(3), (NB[0], DT[0], Rbc[0]), width)
plt.ylabel('accuracy')
print("naive bayes precision = ", NB[0], "recall =", NB[1], "f_measure =", NB[2] )
print("Decision Tree precision = ", DT[0], "recall =", DT[1], "f_measure =", DT[2] )
print("rule based classification precision = ", Rbc[0], "recall =", Rbc[1], "f_measure =", Rbc[2] )

plt.xticks(np.arange(3), ('NB', 'DT', 'RBC'))
plt.yticks(np.arange(0, 101, 10))
plt.show()
