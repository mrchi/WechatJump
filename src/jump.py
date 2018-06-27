#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math

import cv2
import numpy

from adb import PyADB


START_BTN_POS = (0.5, 0.67)
AGAIN_BTN_POS = (0.62, 0.79)
PIECE_IMG_PATH = os.path.join(os.path.dirname(__file__), "../img/piece.png")

class WechatJump:
    def __init__(self, device_serial):
        self.adb = PyADB(device_serial)
        self.resolution = self.adb.get_resolution()
        self.start_btn = [int(v*k) for v, k in zip(self.resolution, START_BTN_POS)]
        self.again_btn = [int(v*k) for v, k in zip(self.resolution, AGAIN_BTN_POS)]
        self.piece = cv2.imread(PIECE_IMG_PATH, cv2.IMREAD_GRAYSCALE)

    def start_game(self):
        """点击开始游戏按钮"""
        self.adb.short_tap(self.start_btn)

    def another_game(self):
        """点击再玩一局按钮"""
        self.adb.short_tap(self.again_btn)

    def find_piece(self, img):
        """
        使用模版匹配寻找棋子位置。

        必须使用当前分辨率下的棋子图片作为模版，否则模版与当前棋子大小不一致时匹配结果很差。
        """
        # 当前分辨率下棋子模版中心点坐标偏移量
        x_delta = 38
        y_delta = 186
        result = cv2.matchTemplate(img, self.piece, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.8:
            return (maxLoc[0]+x_delta, maxLoc[1]+y_delta)
        else:
            return None

    def find_target_center(self, img):
        """
        使用边缘检测寻找下一个落脚点中心位置。

        灰度图像 -> 高斯模糊 -> Canny边缘检测。
        """
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 1, 10)

        # 从 H/3 的位置开始遍历，避免分数和右上角小程序按钮的影响
        y_delta = self.resolution[1]//3
        # 上顶点的坐标
        y_top = numpy.nonzero([max(row) for row in img[y_delta:]])[0][0] + y_delta
        x = int(numpy.mean(numpy.nonzero(img[y_top])))
        # 下顶点的y坐标
        for y in range(y_top+10, self.resolution[1]*2//3):
            if img[y, x] != 0:
                y_bottom = y
                break
        else:
            return None

        return x, (y_top + y_bottom) // 2

    def run(self):
        while True:
            # 读取图片
            img = self.adb.screencap()
            img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2GRAY)
            piece_position = self.find_piece(img)
            if not piece_position:
                break
            target_position = self.find_target_center(img)
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(piece_position, target_position)))
            k = 1.36
            self.adb.long_tap([i//2 for i in self.resolution], int(distance*k))
            time.sleep(0.5)


if __name__ == '__main__':
    wj = WechatJump("48a666d9")
    wj.run()
