import numpy as np

# 多变量高斯分布
from scipy.stats import multivariate_normal

# 用于生成聚类数据集
from sklearn.datasets import make_blobs

# macOS上的matplotlib库导入需要使用下面两句
import matplotlib
matplotlib.use('TkAgg')

# 导入matplotlib库，用于展示聚类数据点
import matplotlib.pyplot as plt

# EM模型类
class EM(object):
    def __init__(self, model):
        '''
        类的构造函数，给模型命名
        model: 模型名称
        '''
        self.__modelName__ = model

    def init_params(self, shape, K):
        '''
        初始化各参数，包括均值、协方差矩阵和每个高斯分布密度函数的权重
        shape: 聚类数据集的维度，包括数据量和数据维度
        K: 聚类的目标类别数量
        '''
        N, D = shape
        mu = np.random.rand(K, D)
        cov = np.array([np.eye(D)] * K)
        alpha = np.array([1.0 / K] * K)
        return mu, cov, alpha

    def E_step(self, Y, mu, cov, alpha):
        '''
        E步骤，求均值，
        Y: 输入数据集
        mu: 均值
        cov: 协方差矩阵
        alpha: 各高斯分布密度函数的权重
        返回隐变量gamma
        '''
        N = Y.shape[0]
        K = alpha.shape[0]

        gamma = np.mat(np.zeros((N, K)))

        prob = np.zeros((N, K))
        for k in range(K):
            prob[:, k] = self.phi(Y, mu[k], cov[k])
        prob = np.mat(prob)

        for k in range(K):
            gamma[:, k] = alpha[k] * prob[:, k]
        for i in range(N):
            gamma[i, :] /= np.sum(gamma[i, :])
        return gamma

    def M_step(self, Y, gamma):
        '''
        M步骤，优化参数，
        Y: 输入数据集
        gamma: 隐变量
        返回优化后的参数
        '''
        N, D = Y.shape
        K = gamma.shape[1]

        mu = np.zeros((K, D))
        cov = []
        alpha = np.zeros(K)

        for k in range(K):
            Nk = np.sum(gamma[:, k])
            mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
            cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
            cov.append(cov_k)
            alpha[k] = Nk / N
        cov = np.array(cov)
        return mu, cov, alpha

    def phi(self, Y, mu_k, cov_k):
        '''
        高斯分布概率密度函数
        Y: 输入变量
        mu_k: 均值
        cov_k: 协方差矩阵
        '''
        norm = multivariate_normal(mean=mu_k, cov=cov_k)
        return norm.pdf(Y)

    def estimate(self, Y, K, times):
        '''
        EM迭代算法，返回最后优化得到的均值、协方差矩阵和各高斯分布密度函数的权重
        Y: 输入数据
        K: 聚类的目标类别数量
        times: 迭代次数
        '''
        mu, cov, alpha = self.init_params(Y.shape, K)
        for i in range(times):
            gamma = self.E_step(Y, mu, cov, alpha)
            mu, cov, alpha = self.M_step(Y, gamma)
        return mu, cov, alpha


if __name__ == '__main__':
    # 100个聚类样本
    # 3个聚类中心
    # 8次EM迭代
    EM_times = 8
    class_num = 3
    sample_num = 100

    # 生成数据集X和标签y，无监督学习不需要使用标签y，X中数据维度为2
    # 建立EM对象，命名为‘GMM’
    # 迭代优化后根据隐变量分出类别
    X, y = make_blobs(n_samples=sample_num, n_features=2, centers=class_num)
    GMM_EM = EM('GMM')
    mu, cov, alpha = GMM_EM.estimate(Y=X, K=class_num, times=EM_times)
    gamma = GMM_EM.E_step(X, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]

    # 按类别整理数据点，并画图显示
    c0 = np.array([X[i] for i in range(sample_num) if category[i] == 0])
    c1 = np.array([X[i] for i in range(sample_num) if category[i] == 1])
    c2 = np.array([X[i] for i in range(sample_num) if category[i] == 2])

    plt.plot(c0[:, 0], c0[:, 1], 'rs', label="class0")
    plt.plot(c1[:, 0], c1[:, 1], 'bo', label="class1")
    plt.plot(c2[:, 0], c2[:, 1], 'gv', label="class2")
    plt.legend(loc="best")
    plt.title("GMM Clustering By EM Algorithm")
    plt.show()
