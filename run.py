#!/usr/bin/env python3
# coding=utf-8

from src.model import MachineLearningModel
from src.jump import WechatJump
from src.adb import PyADB

# 请根据实际情况修改以下配置项
# ------------------------------------------------------------------------------
ADB_DEVICE_SERIAL = "48a666d9"              # adb devices 命令得到的手机序列号

TRAINING_DATASET = "./training.txt"         # 训练数据文件路径
TRAINING_MODEL = "LR"                       # LR 线性回归模型，PR 一元多项式回归模型
TRAINING_PR_DEGREE = 3                      # 一元多项式回归模型时，变量的最高指数

JUMP_DELAY = 1.1                            # 跳跃后与下次截图之间的时间间隔，单位秒
SHOW_MARKED_IMG = False                     # 展示标注数据的 RGB 图像
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # adb 初始化
    adb = PyADB(ADB_DEVICE_SERIAL)

    # 训练回归模型
    model = MachineLearningModel(TRAINING_DATASET)
    if TRAINING_MODEL == "LR":
        model.train_linear_regression_model()
    elif TRAINING_MODEL == "PR":
        model.train_polynomial_regression_model(degree=TRAINING_PR_DEGREE)

    # 跳一跳类初始化
    wj = WechatJump(adb, model)

    # 运行
    wj.run(jump_delay=JUMP_DELAY, show_img=SHOW_MARKED_IMG)
