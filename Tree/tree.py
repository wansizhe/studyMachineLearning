import operator
from math import log
from collections import Counter
import random
import sklearnDT
import numpy as np
from sklearn.model_selection import train_test_split

def createDataSet():
    """
    创建数据集。

    Args:
        None

    Returns:
        dataSet (list)  数据集
        labels  (list)  标签集，表达特征的意义
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [1, 0, 'yes'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['A', 'B']
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算香农熵

    Args:
        dataSet (list)  数据集

    Returns:
        entropy (float) 熵值
    """
    numEntries = len(dataSet)
    
    # labelCounts = {}
    # # 统计标签
    # # 等价于下方 labelCounts = Counter(...)
    # for featVec in dataSet:
    #     currLabel = featVec[-1]
    #     if currLabel not in labelCounts.key():
    #         labelCounts[currLabel] = 0
    #     labelCounts[currLabel] += 1
    # # 计算熵
    # # 等价于下方 probs = [...]
    # #          shannonEnt = sum(...)
    # shannonEnt = 0.0
    # for key in labelCounts:
    #     prob = float(labelCounts[key]) / numEntries
    #     shannonEnt -= prob * log(prob, 2)
    
    labelCounts = Counter(data[-1] for data in dataSet)
    probs = [p[1] / len(dataSet) for p in labelCounts.items()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])

    return shannonEnt

def splitDataSet(dataSet, index, value):
    """
    分割数据集，在index对应的feature里，划分出值为value的行，最终要删除index对应的特征

    Args:
        dataSet (list)  数据集
        index   (int)   所选特征
        value   (int)   feature的取值

    Returns:
        subDataSet  (list)  划分出的子数据集   
    """
    subDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            # 删除index列
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            subDataSet.append(reducedFeatVec)
    return subDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的分类特征

    Args:
        dataSet (list) 数据集

    Returns:
        bestFeature (int)   最优特征的index
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
        # print('infoGain =', infoGain, 'currFeature =', i, 'bestFeature =', bestFeature)

    return bestFeature


def majorityCnt(classList):
    """
    选择一个出现最多的结。
    如果所有的特征都已经用完了，仍然有叶子结点里包含不同label的样本，按照投票少数服从多数。
    
    Args: 
        classList   (list)  所有的label集合
    
    Returns:
        bestFeature (int)   出现最多的label的index
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print('sortedClassCount: ', sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    建立决策树。

    Args: 
        dataSet (list)
        labels  (list)

    Returns: 
        myDecisionTree  (Dict)
    """
    classList = [example[-1] for example in dataSet]
    createLabels = labels[:]

    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = createLabels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(createLabels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = createLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLables, testVec):
    """
    为新样本分类

    Args:
        inputTree   (dict)  树模型
        featLabels  (list)  特征名称
        testVec     (list)  新样本

    Returns:
        classLabel  (str)   分类结果   
    """
    # 获取根结点的值（某特征），和对应子树
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 找到根结点的值（某特征）在label集合中的index
    featIndex = featLables.index(firstStr)

    # 找到测试样本中对应的特征值
    key = testVec[featIndex]
    # 走到子树中的对应分支
    valueOfFeat = secondDict[key]
    print('+++', firstStr, '---', key, '>>>', valueOfFeat)

    # 递归继续分类
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLables, testVec)
    else:
        classLabel = valueOfFeat

    return classLabel

def testTreeClassify():
    data, labels = createDataSet()
    rate = 0.1

    m = len(data)
    numTestVec = int(m * rate)
    testData = random.sample(data, numTestVec)
    aData = list(data)
    bData = list(testData)
    
    for i in bData:
        if i in aData:
            aData.remove(i)
    trainData = aData

    inputTree = createTree(trainData, labels)
    print(inputTree)
    print(labels)

    error = 0.0
    for i in range(numTestVec):
        testVec = testData[i][: -1]
        classLabel = classify(inputTree, labels, testVec)
        if classLabel != testData[i][-1]:
            error += 1
            print(testVec, testData[i][-1], classLabel)

    print("error:", error / numTestVec)
    


if __name__ == "__main__":
    iris = sklearnDT.createDataSet()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    labels = iris.feature_names
    dataset = np.hstack((X_train, y_train.reshape(y_train.size, 1))).tolist()
    mytree = createTree(dataset, labels)
    print(mytree)
    print(labels)
    print('\n')
    
    error = 0.0
    for i in range(y_test.size):
        pred = classify(mytree, labels, X_test[i].tolist())
        truth = y_test[i]
        print('prediction = %s, truth = %s' % (labels[pred], labels[truth]))
        if pred != truth:
            error += 1
    print('error = %f' % error / y_test.size)
