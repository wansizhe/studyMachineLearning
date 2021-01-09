import numpy as np 
import operator


def createDatasets():
	group = np.array([[1, 101], 
					  [5, 89], 
					  [108, 5], 
					  [115, 8]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):
	'''
	inX: new data without label
	dataSet: data for training
	labels: labels for training
	k: k for kNN
	'''
	# 将新数据重复扩展，和训练数据同维度
	# 分别计算新数据点到各个训练数据点的欧式距离
	# 按照欧式距离值进行排序
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistances = distances.argsort()
	# 用前k个最近的数据点的label进行投票
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistances[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 根据投票结果，即字典的value项反向排序，返回第一个
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


if __name__ == '__main__':
	group, labels = createDatasets()
	test = [101, 20]
	test_class = classify0(test, group, labels, 3)
	print(test_class)
