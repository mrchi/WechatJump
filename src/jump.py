#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math

import cv2
import numpy as np
from PIL import ImageDraw

from adb import PyADB

NULL_POS = np.array([0, 0])


class WechatJump:
    """ 所有的坐标都是 (x, y) 格式的，但是在 opencv 的数组中是 (y, x) 格式的"""
    def __init__(self, device_serial):
        self.adb = PyADB(device_serial)
        self.resolution = np.array(self.adb.get_resolution())
        self.start_btn = self.resolution * np.array([0.5, 0.67])
        self.again_btn = self.resolution * np.array([0.62, 0.79])
        self.piece = cv2.imread("../img/piece.png", cv2.IMREAD_GRAYSCALE)
        self.piece_delta = np.array([38, 186])
        self.center = cv2.imread("../img/center.png", cv2.IMREAD_GRAYSCALE)
        self.center_delta = np.array([19, 15])

    def start_game(self):
        """点击开始游戏按钮"""
        self.adb.short_tap(self.start_btn)

    def another_game(self):
        """点击再玩一局按钮"""
        self.adb.short_tap(self.again_btn)

    @staticmethod
    def match_template(img, tpl, threshold=0.8, debug=False):
        """opencv模版匹配，图像要先处理为灰度图像"""
        result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        return np.array(maxLoc) if maxVal >= threshold else NULL_POS

    def get_piece_pos(self, img):
        """
        使用模版匹配寻找棋子位置。

        必须使用当前分辨率下的棋子图片作为模版，否则模版与当前棋子大小不一致时匹配结果很差。
        """
        match_pos = self.match_template(img, self.piece)
        if not match_pos.any():
            raise ValueError("无法定位棋子")
        self.piece_pos =  match_pos + self.piece_delta
        return self.piece_pos

    def match_center_tpl(self, img):
        """使用模版匹配寻找小白点，小白点在跳中棋盘中心后出现。"""
        match_pos = self.match_template(img, self.center)
        if match_pos.any():
            self.target_pos = match_pos + self.center_delta
        else:
            self.target_pos = NULL_POS
        return self.target_pos

    def get_target_img(self, img):
        """获取当前目标棋盘的图像。"""
        half_height = self.target_pos[1] - self.top_pos[1]
        # 0.57735 是 tan 30 的约数
        half_width = int(half_height/0.57735)
        self.target_img = img[
            self.target_pos[1]: self.target_pos[1]+half_height+100,
            self.target_pos[0]-half_width: self.target_pos[0]+half_width,
        ]
        return self.target_img

    def get_target_pos(self, img):
        """
        获取目标棋盘中心点坐标。

        1. 使用模版匹配寻找小白点。
        2. 使用 Canny 边缘检测寻找目标棋盘上顶点坐标，边缘检测：灰度图像 -> 高斯模糊
         -> Canny边缘检测。
        3. 如果模版匹配没有找到小白点，则寻找下顶点并计算目标

        """
        self.match_center_tpl(img)

        # 高斯模糊后，处理成Canny边缘图像
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 1, 10)

        # 有时棋子高度高于落脚点，为去掉棋子对判断的影响，抹掉棋子的边缘，将像素值置为0
        # 这里数组的索引是 img[y1:y2, x1:x2] 的形式
        img[
            self.piece_pos[1]-self.piece_delta[1]:self.piece_pos[1]+2,
            self.piece_pos[0]-self.piece_delta[0]:self.piece_pos[0]+self.piece_delta[0],
        ] = 0

        # 为避免屏幕上半部分分数和小程序按钮的影响
        # 从 1/3*H 的位置开始向下逐行遍历到 2/3*H，寻找目标棋盘的上顶点
        y_start = self.resolution[1] // 3
        y_stop = self.resolution[1] // 3 * 2

        # 上顶点的 y 坐标
        for y in range(y_start, y_stop):
            if img[y].any():
                y_top = y
                break
        else:
            raise ValueError("无法定位目标棋盘上顶点")

        # 上顶点的 x 坐标，也是中心点的 x 坐标
        x = int(np.mean(np.nonzero(img[y_top])))
        self.top_pos = np.array([x, y_top])

        # 如果模版匹配已经找到了目标棋盘中心点，就不需要继续操作了。
        if self.target_pos.any():
            return self.target_pos

        # 下顶点的 y 坐标，+130 是为了消除多圆环类棋盘的干扰
        for y in range(y_top+130, y_stop):
            if img[y, x] or img[y, x-1]:
                y_bottom = y
                break
        else:
            raise ValueError("无法定位目标棋盘下顶点")

        # 由上下顶点 y 坐标获得中心点 y 坐标
        self.target_pos = np.array([x, (y_top + y_bottom) // 2])
        return self.target_pos

    def get_start_pos(self, img):
        """通过模版匹配，获取起始棋盘中心坐标"""
        self.start_pos = NULL_POS
        if hasattr(self, "target_img"):
            match_pos = self.match_template(img, self.target_img, 0.7)
            if match_pos.any():
                shape = self.target_img.shape
                start_pos = match_pos + np.array([shape[1]//2, 0])
                # 如果坐标与当前棋子坐标差距过大，则认为有问题，丢弃
                if (np.abs(start_pos-self.piece_pos) < np.array([100, 100])).all():
                    self.start_pos = start_pos
        return self.start_pos

    def run(self):
        while True:
            # 读取图片
            img_rgb = self.adb.screencap()
            img = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2GRAY)
            self.get_piece_pos(img)
            self.get_target_pos(img)
            self.get_start_pos(img)
            self.get_target_img(img)
            self.show_img(img_rgb)
            distance = math.sqrt(
                sum(
                    (a-b)**2 for a, b in zip(self.piece_pos, self.target_pos)
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
            (0, self.piece_pos[1], self.resolution[0], self.piece_pos[1]),
            "#ff0000",
        )
        draw.line(
            (self.piece_pos[0], 0, self.piece_pos[0], self.resolution[1]),
            "#ff0000",
        )
        # 目标棋盘中心点
        draw.line(
            (0, self.target_pos[1], self.resolution[0], self.target_pos[1]),
            "#0000ff",
        )
        draw.line(
            (self.target_pos[0], 0, self.target_pos[0], self.resolution[1]),
            "#0000ff",
        )
        # 当前棋盘中心点
        draw.line(
            (0, self.start_pos[1], self.resolution[0], self.start_pos[1]),
            "#000000",
        )
        draw.line(
            (self.start_pos[0], 0, self.start_pos[0], self.resolution[1]),
            "#000000",
        )
        img_rgb.show()


if __name__ == '__main__':
    wj = WechatJump("48a666d9")
    wj.run()
