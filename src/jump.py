#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

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

    def run(self):
        # 读取图片
        img = self.adb.screencap()
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2GRAY)
        piece_pos = self.find_piece_position(img)
        print(piece_pos)

    def find_piece_position(self, img):
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
# 循环中
    # 截图
    # 判断游戏结束
    # 获取小人位置
    # 获取方块和下一个方块位置
    # 计算按压时间
    # 按压操作


if __name__ == '__main__':
    wj = WechatJump("48a666d9")
    wj.run()
