# coding=utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

__all__ = ["MachineLearningModel"]


class MachineLearningModel:
    """机器学习回归分析"""
    def __init__(self, dataset_file, only_center=False):
        self.read_training_datasets(dataset_file, only_center)

    def read_training_datasets(self, dataset_file, only_center=False):
        """读取训练数据"""
        with open(dataset_file, "r") as f:
            data = f.readlines()

        dataset_X = []
        dataset_Y = []

        for line in data:
            line = line.strip()
            if not line:
                continue
            x, y, z = line.split()
            # 只取跳中了中心的数据
            if only_center and z == "False":
                continue
            dataset_X.append(float(x))
            dataset_Y.append(int(y))

        self.dataset_X = np.array(dataset_X).reshape([len(dataset_X), 1])
        self.dataset_Y = np.array(dataset_Y)

    def train_linear_regression_model(self):
        """训练线性回归模型"""
        linear = LinearRegression()
        linear.fit(self.dataset_X, self.dataset_Y)
        self.predict = lambda x: linear.predict(x)[0]

    def train_polynomial_regression_model(self, degree):
        """训练多项式回归模型"""
        poly_feat = PolynomialFeatures(degree=degree)
        x_tranformed = poly_feat.fit_transform(self.dataset_X)
        linear = LinearRegression()
        linear.fit(x_tranformed, self.dataset_Y)
        self.predict = lambda x: linear.predict(poly_feat.transform(x))[0]

