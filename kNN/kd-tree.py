import numpy as np
import time

class Node(object):

    def __init__(self, item=None, label=None, dim=None, parent=None, left=None, right=None):
        self.item = item
        self.label = label
        self.dim = dim
        self.parent = parent
        self.left = left
        self.right = right


class KDTree(object):

    def __init__(self, aList, labelList):
        self.__length = 0
        self.__root = self.__create(aList, labelList)
