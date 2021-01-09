import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class ContinuousNaiveBayesClassifier(object):
    '''
    连续型朴素贝叶斯分类器
    '''
    
    def __init__(self, train_data, train_label):
        '''
        初始化类
        train_data: 训练数据
        train_label: 训练数据分类标签
        '''
        self.data = train_data
        self.label = train_label
        self.__lambda__ = 1
        self.statistics()

    def statistics(self):
        '''
        统计训练数据集，包括
        label_statistics: 每种类别中包含的数据量
        K: 类别的数量
        N: 数据量
        mean: 每一类数据在所有特征上的均值
        std: 每一类数据在特征上的标准差
        '''
        self.label_statistics = Counter(self.label).items()
        self.K = len(self.label_statistics)
        self.N, self.num_feature = self.data.shape
        self.mean = {}
        self.std = {}
        for cla, num in self.label_statistics:
            curr_data = self.mask(cla)
            self.mean[cla] = np.mean(curr_data, axis=0)
            self.std[cla] = np.std(curr_data, axis=0)

    def normal_distribution_prob(self, x, mean, stdev):
        '''
        正态分布概率密度函数
        x: 样本数据
        mean: 均值
        stdev: 标准差
        '''
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * np.exp(-(np.power(x - mean, 2)) / (2 * np.power(stdev, 2)))

    def prior(self):
        '''
        计算每一类的先验概率
        __lambda__: 拉普拉斯平滑参数，设定为1
        prior_prob: 字典类型，包含每一类的先验概率
        '''
        prior_prob = {}
        for cla, num in self.label_statistics:
            prior_prob[cla] = (num + self.__lambda__) / (self.N + self.K * self.__lambda__)
        return prior_prob

    def mask(self, label):
        '''
        按类筛选样本
        label: 类别标签
        '''
        return self.data[self.label == label]

    def classify(self, x):
        '''
        为一个样本分类
        x: 样本特征向量
        返回类别标签
        '''
        prior_prob = self.prior()
        # print(prior_prob)
        cla_prob = {}
        for cla, num in self.label_statistics:
            likelihood = np.prod(self.normal_distribution_prob(x, self.mean[cla], self.std[cla]))
            cla_prob[cla] = likelihood * prior_prob[cla]
        return max(cla_prob, key=lambda x: cla_prob[x])

    def predict(self, test_data):
        '''
        预测一组数据的分类
        test_data: 测试数据集
        返回一组类别标签，和测试数据集一一对应
        '''
        prediction = []
        for each in test_data:
            prediction.append(self.classify(each))
        return np.array(prediction)

    def accuracy(self, prediction, target):
        '''
        计算准确率
        prediction: 一组数据的预测类别
        target: 一组数据的真实类别
        '''
        return np.sum(prediction == target) / target.size


if __name__ == '__main__':
    iris = load_iris()  # 加载鸢尾花数据集
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)  # 按照8:2的比例划分训练集和测试集
    classifier = ContinuousNaiveBayesClassifier(X_train, y_train)  # 实例化一个连续型朴素贝叶斯分类器的对象，传入训练数据
    prediction = classifier.predict(X_test)  # 为测试集预测分类结果
    print(prediction)  # 打印测试集分类结果
    acc = classifier.accuracy(prediction, y_test)  # 计算预测准确率
    print('acc: %5f' % acc)  # 打印准确率
