import numpy as np
from collections import defaultdict

class MaxEntropy:

    def __init__(self, trainDataList, trainLabelList, testDataList, testLabelList):
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self,testDataList = testDataList
        self.trainLabelList = testLabelList
        self.featureNum = len(trainDataList[0])

        self.N = len(trainDataList)
        self.n = 0
        self.M = 10000
        self.fixy = self.calc_fixy()
        self.w = [0] * self.n
        self.xy2idDict, self.id2xyDict = self.createSearchDict()
        self.Ep_xy = self.calcEp_xy()

    def calcEpxy(self):
        Epxy = [0] * self.n
        for i in range(self.N):
            Pwxy = [0] * 2
            Pwxy[0] = self.calcPwy_x(self.trainDataList[i], 0)
            Pwxy[1] = self.calcPwy_x(self.trainDataList[i], 1)

            for feature in range(self.featureNum):
                for y in range(2):
                    if (self.trainDataList[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy

    def calcEp_xy(self):
        Ep_xy = [0] * self.n

        for feature in range(self.featureNum):
            for (x, y) in self.fixy[feature]:
                id = self.xy2idDict[feature][(x, y)]
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N
        return Ep_xy