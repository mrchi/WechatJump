# coding=utf-8

import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt


class MachineLearningModel:
    """机器学习回归分析"""
    def __init__(self):
        self.read_training_datasets()

    def read_training_datasets(self):
        """读取训练数据"""
        with open("./training.txt", "r") as f:
            data = f.readlines()

        datasets_X = []
        datasets_Y = []

        for line in data:
            line = line.strip()
            if not line:
                continue
            x, y = line.split()
            datasets_X.append(float(x))
            datasets_Y.append(int(y))

        self.datasets_X = np.array(datasets_X).reshape([len(datasets_X), 1])
        self.datasets_Y = np.array(datasets_Y)

    def train_linear_regression_model(self):
        """训练线性回归模型"""
        linear = linear_model.LinearRegression()
        linear.fit(self.datasets_X, self.datasets_Y)
        self.predict = lambda x: linear.predict(x)[0]

    def train_polynomial_regression_model(self, degree=5):
        """训练多项式回归模型"""
        poly_feat = preprocessing.PolynomialFeatures(degree=degree)
        x_tranformed = poly_feat.fit_transform(self.datasets_X)
        linear = linear_model.LinearRegression()
        linear.fit(x_tranformed, self.datasets_Y)
        self.predict = lambda x: linear.predict(poly_feat.transform(x))[0]

