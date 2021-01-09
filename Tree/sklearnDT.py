from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pydotplus

def createDataSet():
    iris = load_iris()
    """
    iris.data
    iris.target
    iris.target_names
    iris.feature_names
    """
    # pd.concat([pd.DataFrame(iris.data), pd.DataFrame(iris.target)], axis=1)
    return iris


def createTree(X_train, y_train):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    return clf

def predictByTree(clf, X_test, y_test):
    y_pre = clf.predict(X_test)
    acc = np.mean(y_pre == y_test)
    return y_pre, acc

def showPR(dataSet, clf, y_train, y_pre):
    # 二分类使用
    precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
    answer = clf.predict_proba(dataSet.data)[:, 1]
    print(classification_report(dataSet.target, answer, dataSet.feature_names))
    print(answer)

def showPic(clf, dataSet, path='./tree.pdf'):
    pic = tree.export_graphviz(
        clf,
        feature_names=dataSet.feature_names,
        class_names=dataSet.target_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = pydotplus.graph_from_dot_data(pic)
    graph.write_pdf(path)


if __name__ == "__main__":
    iris = createDataSet()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    clf = createTree(X_train, y_train)
    y_pre, acc = predictByTree(clf, X_test, y_test)
    print("acc = %f" % acc)

    # showPR(iris, clf, y_train, y_pre)
    showPic(clf, iris)