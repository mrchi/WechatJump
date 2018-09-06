#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math

import cv2
import numpy
from PIL import ImageDraw

from adb import PyADB


START_BTN_POS = (0.5, 0.67)
AGAIN_BTN_POS = (0.62, 0.79)
IMG_PATH = os.path.join(os.path.dirname(__file__), "../img")
PIECE_IMG_PATH = os.path.join(IMG_PATH, "piece.png")
CENTER_IMG_PATH = os.path.join(IMG_PATH, "center.png")


class WechatJump:
    def __init__(self, device_serial):
        self.adb = PyADB(device_serial)
        self.resolution = self.adb.get_resolution()
        self.start_btn = [int(v*k) for v, k in zip(self.resolution, START_BTN_POS)]
        self.again_btn = [int(v*k) for v, k in zip(self.resolution, AGAIN_BTN_POS)]
        self.piece = cv2.imread(PIECE_IMG_PATH, cv2.IMREAD_GRAYSCALE)
        self.piece_delta = (38, 186)
        self.center = cv2.imread(CENTER_IMG_PATH, cv2.IMREAD_GRAYSCALE)
        self.center_delta = (19, 15)

    def start_game(self):
        """点击开始游戏按钮"""
        self.adb.short_tap(self.start_btn)

    def another_game(self):
        """点击再玩一局按钮"""
        self.adb.short_tap(self.again_btn)

    def _match_template(self, img, tpl, threshold=0.8):
        """opencv模版匹配，图像要先处理为灰度图像"""
        result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        return maxLoc if maxVal >= threshold else None

    def find_piece(self, img):
        """
        使用模版匹配寻找棋子位置。

        必须使用当前分辨率下的棋子图片作为模版，否则模版与当前棋子大小不一致时匹配结果很差。
        """
        match_pos = self._match_template(img, self.piece)
        if match_pos:
            return (match_pos[0]+self.piece_delta[0], match_pos[1]+self.piece_delta[1])
        else:
            return None

    def find_target_center(self, img):
        """
        先使用模版匹配寻找小白点，如果没有找到，再使用边缘检测寻找下一个落脚点中心位置。

        边缘检测：灰度图像 -> 高斯模糊 -> Canny边缘检测。
        """
        match_pos = self._match_template(img, self.center, 0.85)
        if match_pos:
            return (match_pos[0]+self.center_delta[0], match_pos[1]+self.center_delta[1])

        # 边缘检测
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 1, 10)

        # 有时棋子高度高于落脚点，去掉棋子对判断的影响
        for y in range(
                    self.piece_position[1]-self.piece_delta[1],
                    self.piece_position[1]+2,
                ):
            for x in range(
                        self.piece_position[0]-self.piece_delta[0],
                        self.piece_position[0]+self.piece_delta[0],
                    ):
                img[y][x] = 0

        # 从 H/3 的位置开始遍历，避免分数和右上角小程序按钮的影响
        y_delta = self.resolution[1]//3
        # 上顶点的坐标
        y_top = numpy.nonzero([max(row) for row in img[y_delta:]])[0][0] + y_delta
        x = int(numpy.mean(numpy.nonzero(img[y_top])))
        # 下顶点的y坐标
        for y in range(y_top+130, self.resolution[1]*2//3):
            if img[y, x] != 0 or img[y, x-1] != 0:
                y_bottom = y
                break
        else:
            return None

        return x, (y_top + y_bottom) // 2

    def run(self):
        while True:
            # 读取图片
            img_rgb = self.adb.screencap()
            img = cv2.cvtColor(numpy.asarray(img_rgb), cv2.COLOR_RGB2GRAY)
            self.piece_position = self.find_piece(img)
            if not self.piece_position:
                print("无法定位棋子")
                break
            self.target_position = self.find_target_center(img)
            if not self.target_position:
                print("无法定位落脚点")
                break
            self.show_img(img_rgb)
            distance = math.sqrt(
                sum(
                    (a-b)**2 for a, b in zip(self.piece_position, self.target_position)
                )
            )
            k = 1.365
            duration = max(int(distance*k), 300)
            self.adb.long_tap([i//2 for i in self.resolution], duration)
            time.sleep(1.2)

    def show_img(self, img_rgb):
        draw = ImageDraw.Draw(img_rgb)
        # 棋子中心点
        draw.line(
            (0, self.piece_position[1], self.resolution[0], self.piece_position[1]),
            "#ff0000",
        )
        draw.line(
            (self.piece_position[0], 0, self.piece_position[0], self.resolution[1]),
            "#ff0000",
        )
        # 落脚点中心点
        draw.line(
            (0, self.target_position[1], self.resolution[0], self.target_position[1]),
            "#0000ff",
        )
        draw.line(
            (self.target_position[0], 0, self.target_position[0], self.resolution[1]),
            "#0000ff",
        )
        img_rgb.show()


if __name__ == '__main__':
    wj = WechatJump("48a666d9")
    wj.run()
