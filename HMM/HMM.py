import numpy as np
import csv
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class HMM(object):
    def __init__(self, N, M):
        """
        N: 隐状态变量集合大小 q1, q2, ..., qN
        M: 观测变量集合大小 v1, v2, ..., vM
        A: 状态转移矩阵
        B: 发射矩阵｜观测概率矩阵
        Pi: 初始概率分布
        """
        self.A = np.zeros((N, N))
        self.B = np.zeros((N, M))
        self.Pi = np.array([1.0 / N] * N)
        self.N = N
        self.M = M

    def cal_prob(self, O):
        '''
        计算 P(O|lambda)
        T: 序列长度
        O: 观测序列
        '''
        self.T = len(O)
        self.O = O
        self.forward()
        return sum(self.alpha[self.T-1])

        # self.backward()
        # sum = 0
        # for i in range(N):
        #     sum += self.Pi[i] * self.B[i][self.O[0]] * self.beta[0][i]

    def forward(self):
        '''
        解决Evaluation问题的前向算法
        主要是递推计算alpha
        alpha: T*N维矩阵，alpha[t,k]=P(O[:t],i[t]=qk)|lambda)
        '''
        # forward-1: 初始化
        self.alpha = np.zeros((self.T, self.N))
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.O[0]]
        # forward-2: 递推
        for t in range(1, self.T):
            for i in range(self.N):
                sum = 0
                for j in range(self.N):
                    sum += self.alpha[t-1][j] * self.A[j][i]
                self.alpha[t][i] = sum * self.B[i][self.O[t]]

    def backward(self):
        '''
        解决Evaluation问题的后向算法
        主要是递推计算beta
        beta: T*N维矩阵，beta[t,k]=P(O[t+1:]|i[t]=qk,lambda)
        '''
        # backward-1: 初始化
        self.beta = np.zeros((self.T, self.N))
        for i in range(self.N):
            self.beta[self.T-1][i] = 1
        # backward-2: 递推
        for t in range(self.T-2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]

    def gamma(self, i, t):
        '''
        gamma[t,i]=(alpha[t,i]*beta[t,i)/P(O|lambda)
        '''
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0
        for j in range(self.N):
            denominator += self.alpha[t][j] * self.beta[t][j]
        return numerator / denominator

    def ksi(self, i, j, t):
        '''
        ksi[t,i,j]=P(i[t]=qi,t[t+1]=qj|O,lambda)
                  =P(i[t]=qi,t[t+1]=qj,O|lambda)/P(O|lambda)
                  =alpha[t,i]*A[i,j]*B[j,o[t+1]]*beta[t+1][j]/P(O|lambda)
        '''
        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]
        denominator = 0
        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]
        return numerator / denominator

    def init(self):
        '''
        随机生成 A, B
        每行和为1
        Pi已随机生成好
        '''
        for i in range(self.N):
            randlist = [random.randint(0, 100) for t in range(self.N)]
            s = sum(randlist)
            for j in range(self.N):
                self.A[i][j] = randlist[j] / s
        for i in range(self.N):
            randlist = [random.randint(0, 100) for t in range(self.M)]
            s = sum(randlist)
            for j in range(self.M):
                self.B[i][j] = randlist[j] / s

    def train(self, O, MaxSteps = 100):
        '''
        Baum-Welch / EM 算法
        '''
        self.T = len(O)
        self.O = O

        self.init()
        step = 0;

        while step < MaxSteps:
            step += 1
            print("training...  step %d" % step)
            tmp_A = np.zeros((self.N, self.N))
            tmp_B = np.zeros((self.N, self.M))
            tmp_Pi = np.array([0.0] * self.N)

            self.forward()
            self.backward()

            for i in range(self.N):
                for j in range(self.N):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T-1):
                        numerator += self.ksi(i, j, t)
                        denominator += self.gamma(i, t)
                    tmp_A[i][j] = numerator / denominator

            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T):
                        if k == self.O[t]:
                            numerator += self.gamma(j, t)
                        denominator += self.gamma(j, t)
                    tmp_B[j][k] = numerator / denominator

            for i in range(self.N):
                tmp_Pi[i] = self.gamma(i, 0)

            self.A = tmp_A
            self.B = tmp_B
            self.Pi = tmp_Pi

    def generate(self, length):
        I = []

        rand = random.randint(0, 1000) / 1000.0
        i = 0
        while self.Pi[i] < rand or self.Pi[i] < 0.0001:
            rand -= self.Pi[i]
            i += 1
        I.append(i)

        for t in range(1, length):
            last = I[-1]
            rand = random.randint(0, 1000) / 1000.0
            i = 0
            while self.A[last][i] < rand or self.A[last][i] < 0.0001:
                rand -= self.A[last][i]
                i += 1
            I.append(i)

        Y = []
        for t in range(length):
            i = 0
            rand = random.randint(0, 1000) / 1000.0
            while self.B[I[t]][i] < rand or self.B[I[t]][i] < 0.0001:
                rand -= self.B[I[t]][i]
                i += 1
            Y.append(i)

        return I, Y

    def viterbi(self, O):
        self.T = len(O)
        self.O = O
        delta = np.zeros((self.T, self.N))
        psi = np.zeros((self.T, self.N))
        I = np.array([0] * self.T)
        # 初始化
        for i in range(self.N):
            delta[0][i] = self.Pi[i] * self.B[i][self.O[0]]
            psi[0][i] = 0
        # 递推
        for t in range(1, self.T):
            for i in range(self.N):
                delta[t][i] = np.max(np.multiply(delta[t-1], self.A[:,i])) * self.B[i][self.O[t]]
                psi[t][i] = np.argmax(np.multiply(delta[t-1], self.A[:,i]))
        # 回溯
        I[self.T-1] = np.argmax(delta[self.T-1])
        for t in range(self.T-2, -1, -1):
            I[t] = psi[t+1][I[t+1]]
        return I


def triangle(length):
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x %= 6
        if x <= 3:
            Y.append(x)
        else:
            Y.append(6-x)
    return X, Y

def show_data(x, y, color):
    plt.plot(x, y, color)
    plt.show()


if __name__ == '__main__':
    hmm = HMM(10, 4)
    tri_x, tri_y = triangle(20)
    show_data(tri_x, tri_y, 'g')

    hmm.train(tri_y)
    I, y = hmm.generate(100)
    x = [i for i in range(100)]
    show_data(x, y, 'r')

    print(I)
