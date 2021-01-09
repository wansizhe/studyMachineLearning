import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        初始化分类器个数和学习率
        """
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        """
        初始化各项参数
        """
        self.X = datasets
        self.Y = labels
        # M个数据，每个数据N维，也就是N个特征
        self.M, self.N = datasets.shape
        # 分类器参数集合
        self.clf_sets = []
        # 初始化数据权重
        self.weights = [1.0 / self.M] * self.M
        # 分类器权重
        self.alpha = []

    def _G(self, features, labels):
        """
        训练单个分类器
        """
        # 样本数
        m = len(features)
        # 错误率
        error = 100000.0
        # 最佳分类边界值
        best_v = 0.0
        # 寻找特征的最大值和最小值
        features_min = min(features)
        features_max = max(features)

        # 寻找最佳分类边界值
        # 从最小值开始，每次移动一步，尝试分类
        # 移动步长，就是学习率，由此计算出移动步数
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        print('n_step: {}'.format(n_step))
        
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            # 每移动一步，计算当前分类边界值
            v = features_min + self.learning_rate * i 
            # 分类边界值不是特征取值
            if v not in features:
                # 先设定，当大于边界值时取1，否则取-1
                # 计算分类结果和带权重当分类错误率
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weights_error_positive = sum([self.weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])
                # 再设定，当大于边界值时取-1，否则取1
                # 计算分类结果和带权重当分类错误率
                compare_array_negative = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weights_error_negative = sum([self.weights[k] for k in range(m) if compare_array_negative[k] != labels[k]])

                # 在上面两种设定中选择一种错误率更低的
                # direct表示大于或小于边界值取1，weight_error为当前边界值的错误率，_compare_array为分类结果
                if weights_error_positive < weights_error_negative:
                    weights_error = weights_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weights_error = weights_error_negative
                    _compare_array = compare_array_negative
                    direct = 'negative'
                
                print('v: {} error:{}'.format(v, weights_error))

                # 如果错误率达到已有的最小值
                if weights_error < error:
                    error = weights_error
                    compare_array = _compare_array
                    best_v = v
        
        return best_v, direct, error, compare_array

    def _alpha(self, error):
        """
        计算alpha
        """
        # alpha = 1/2 * log((1-e)/e)
        return 0.5 * np.log((1 - error) / error)

    def _Z(self, a, clf):
        """
        计算更新权重时用到的规范化因子
        """
        # sum(w * exp(-a*y*G))
        return sum([self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) for i in range(self.M)])

    def _W(self, a, clf, Z):
        """
        更新权重
        """
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z 

    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        """
        应用单个分类器，返回1或-1
        """
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        """
        adaboost 训练过程
        """
        # 初始化参数
        self.init_args(X, y)

        # 迭代训练每个基分类器
        # 每个基分类器只用N个特征中对一个来分类，从中挑一个最好的
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None

            # 尝试每个特征进行分类器训练
            for j in range(self.N):
                # 取出当前特征
                features = self.X[:, j]
                # 用带权重的数据训练分类器
                v, direct, error, compare_array = self._G(features, self.Y)

                # 如果错误率比已有记录更低，就更新记录
                if error < best_clf_error:
                    # 最低错误率
                    best_clf_error = error
                    # 最佳分类边界值
                    best_v = v 
                    # 分类取1的方向
                    final_direct = direct
                    # 分类结果
                    clf_result = compare_array
                    # 分类用到的特征
                    axis = j
                
                print('epoch:{}/{} feature:{} error:{} v:{}'.format(epoch+1, self.clf_num, j, error, best_v))
                # 如果错误率为0，就不需要尝试其他特征
                if best_clf_error == 0:
                    break
            
            # 根据当前分类器的错误率计算alpha
            a = self._alpha(best_clf_error)

            # 记录下当前分类器的权重alpha和分类器参数（参数包括分类特征、分类边界值，分类结果）
            self.alpha.append(a)
            self.clf_sets.append((axis, best_v, final_direct))

            # 根据alpha和分类结果，计算归一化因子并更新权重
            Z = self._Z(a, clf_result)
            self._W(a, clf_result, Z)

            print('classifier:{}/{} error:{:.3f} v:{} direct:{} a:{:.5f}'.format(epoch + 1, self.clf_num, error, best_v, final_direct, a))
            print('weight:{}'.format(self.weights))
            print('\n')

        print('alpha:{}'.format(self.alpha))
        print('\n')

    def predict(self, feature):
        """
        用训练好的模型对一个特征来预测
        """
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha * self.G(f_input, clf_v, direct)
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        """
        计算测试集上的正确率
        """
        right_count = 0
        for i in len(X_test):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


if __name__ == "__main__":
    X = np.arange(10).reshape(10, 1)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    clf = AdaBoost(n_estimators=3, learning_rate=0.5)
    clf.fit(X, y)