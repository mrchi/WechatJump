# -*- coding: utf-8 -*-

import os
import time
import math

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

from .adb import PyADB
from .model import MachineLearningModel

__all__ = ["WechatJump"]

ASSESTS_DIR = os.path.join(os.path.dirname(__file__), "../assests")     # assests文件目录
PIECE_IMG = os.path.join(ASSESTS_DIR, "piece.png")                      # 棋子图片
CENTER_BLACK_IMG = os.path.join(ASSESTS_DIR, "center_black.png")        # 黑色背景中心点
CENTER_WHITE_IMG = os.path.join(ASSESTS_DIR, "center_white.png")        # 白色背景中心点
TTF_FONT_FILE = os.path.join(ASSESTS_DIR, "font.ttf")                   # 图片标注使用的ttf字体

NULL_POS = np.array([0, 0])                                             # 无法确定坐标时的默认坐标


class WechatJump:
    """ 所有的坐标都是 (x, y) 格式的，但是在 opencv 的数组中是 (y, x) 格式的"""
    def __init__(self, adb, model):
        self.adb = adb
        self.model = model
        # 获取屏幕分辨率，计算 开始按钮、再玩一局按钮和好友排行榜界面返回按钮 的坐标
        self.resolution = np.array(self.adb.get_resolution())
        self.start_btn = self.resolution * np.array([0.5, 0.67])
        self.again_btn = self.resolution * np.array([0.62, 0.79])
        self.top_chart_back_btn = self.resolution * np.array([0.07, 0.87])
        # 读取棋子图片和中心点图片
        self.piece = cv2.imread(PIECE_IMG, cv2.IMREAD_GRAYSCALE)
        self.center_black = cv2.imread(CENTER_BLACK_IMG, cv2.IMREAD_GRAYSCALE)
        self.center_white = cv2.imread(CENTER_WHITE_IMG, cv2.IMREAD_GRAYSCALE)
        # 设置偏移量，模版匹配时得到的坐标是模版左上角像素的坐标，计算棋子位置和中心点位置时要进行偏移
        self.piece_delta = np.array([38, 186])
        self.center_delta = np.array([19, 15])
        self.init_attrs()

    def start_game(self):
        """点击开始游戏按钮"""
        self.adb.short_tap(self.start_btn)

    def another_game(self):
        """点击再玩一局按钮"""
        self.adb.short_tap(self.top_chart_back_btn)
        self.adb.short_tap(self.again_btn)

    @staticmethod
    def match_template(img, tpl, threshold=0.8):
        """opencv模版匹配，传入图像都为灰度图像。
        要使用当前分辨率下的图片模版（棋子、中心点图片等），否则匹配结果很差。
        """
        result = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        return np.array(maxLoc) if maxVal >= threshold else NULL_POS

    @staticmethod
    def calc_distance(a, b, jump_right):
        """两点在跳跃方向（认为与水平线夹角为 30 度）上投影的距离，比欧式距离精确。"""
        if jump_right:
            distance = abs((a[1]-b[1]) - (a[0]-b[0]) / math.sqrt(3))
        else:
            distance = abs((a[1]-b[1]) + (a[0]-b[0]) / math.sqrt(3))
        # # 欧式距离
        # distance = np.sqrt(np.sum(np.square(a-b)))
        return distance

    def match_center_tpl(self, img):
        """使用模版匹配寻找中心点坐标，中心点在上次跳中棋盘中心后出现在目标棋盘中心。"""
        black_match_pos = self.match_template(img, self.center_black)
        white_match_pos = self.match_template(img, self.center_white)
        if black_match_pos.any():
            self.target_pos = black_match_pos + self.center_delta
            self.on_center = True
        elif white_match_pos.any():
            self.target_pos = white_match_pos + self.center_delta
            self.on_center = True
        else:
            self.target_pos = NULL_POS
            self.on_center = False
        return self.target_pos

    def init_attrs(self):
        """初始化变量"""
        # 测量跳跃距离
        self.last_distance = self.distance if hasattr(self, "distance") else None
        self.distance = None
        # 跳跃时间
        self.last_duration = self.duration if hasattr(self, "duration") else None
        self.duration = None
        # 向右跳跃标志位
        self.last_jump_right = self.jump_right if hasattr(self, "jump_right") else None
        self.jump_right = None
        # 目标棋盘的图像
        self.last_target_img = self.target_img if hasattr(self, "target_img") else NULL_POS
        self.target_img = NULL_POS
        # 棋子坐标、目标棋盘中心坐标，当前棋盘中心坐标，当前棋盘上顶点坐标
        self.piece_pos = NULL_POS
        self.target_pos = NULL_POS
        self.start_pos = NULL_POS
        self.top_pos = NULL_POS
        # 上次跳跃实际距离
        self.last_actual_distance = None
        # 上次跳跃跳中中心标志位
        self.on_center = None

    def get_piece_pos(self, img):
        """使用模版匹配寻找棋子位置，同时判断跳跃方向。"""
        match_pos = self.match_template(img, self.piece, 0.7)
        if not match_pos.any():
            raise ValueError("无法定位棋子")
        self.piece_pos = match_pos + self.piece_delta

        # 计算跳跃方向。若棋子在图片左侧半区，则向右跳跃（不存在在中线上的情况）
        self.jump_right = self.piece_pos[0] < self.resolution[0] // 2

        return self.piece_pos

    def get_target_pos(self, img):
        """
        获取目标棋盘中心点坐标、上顶点坐标。步骤：

        1. 使用模版匹配寻找中心点。
        2. 使用边缘检测寻找目标棋盘上顶点坐标，灰度图像 -> 高斯模糊 -> Canny边缘图像 -> 逐行遍历。
        3. 如果第1步模版匹配没有找到中心点，则寻找目标棋盘下顶点并计算中心点。
        """
        self.match_center_tpl(img)

        # 高斯模糊后，处理成Canny边缘图像
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 1, 10)

        # 有时棋子图像的高度高于目标棋盘，为去掉棋子对判断的影响，抹掉棋子的边缘图像（将像素值置为0）
        # 图像数组的索引是先y后x，即 img[y1:y2, x1:x2] 的形式
        # +2 -2 扩大抹除范围，保证完全抹掉棋子
        img[
            self.piece_pos[1]-self.piece_delta[1]-2: self.piece_pos[1]+2,
            self.piece_pos[0]-self.piece_delta[0]-2: self.piece_pos[0]+self.piece_delta[0]+2,
        ] = 0

        # 为避免屏幕上半部分分数和小程序按钮的影响
        # 从 1/3*H 的位置开始向下逐行遍历到 2/3*H，寻找目标棋盘的上顶点
        y_start = self.resolution[1] // 3
        y_stop = self.resolution[1] * 2 // 3

        # 上顶点的 y 坐标
        for y in range(y_start, y_stop):
            if img[y].any():
                y_top = y
                break
        else:
            raise ValueError("无法定位目标棋盘上顶点")

        # 上顶点的 x 坐标，也是中心点的 x 坐标
        x = int(round(np.mean(np.nonzero(img[y_top]))))
        self.top_pos = np.array([x, y_top])

        # 如果模版匹配已经找到了目标棋盘中心点，就不需要再继续寻找下顶点继而确定中心点
        if self.target_pos.any():
            return self.target_pos

        # 下顶点的 y 坐标，+40是为了消除多圆环类棋盘的干扰
        for y in range(y_top+40, y_stop):
            if img[y, x] or img[y, x-1]:
                y_bottom = y
                break
        else:
            raise ValueError("无法定位目标棋盘下顶点")

        # 由上下顶点 y 坐标取中点获得中心点 y 坐标
        self.target_pos = np.array([x, (y_top + y_bottom) // 2])
        return self.target_pos

    def get_start_pos(self, img):
        """通过模版匹配，获取起始棋盘中心坐标"""
        # 上次跳跃截图的目标棋盘，就是本次跳跃的起始棋盘（棋子所在的棋盘）
        if self.last_target_img.any():
            match_pos = self.match_template(img, self.last_target_img, 0.7)
            if match_pos.any():
                shape = self.last_target_img.shape
                start_pos = match_pos + np.array([shape[1]//2, 0])
                # 如果起始棋盘坐标与当前棋子坐标差距过大，则认为有问题，丢弃
                if (np.abs(start_pos-self.piece_pos) < np.array([100, 100])).all():
                    self.start_pos = start_pos
        return self.start_pos

    def review_last_jump(self):
        """评估上次跳跃参数，计算实际跳跃距离。"""
        # 如果这些属性不存在，就无法进行评估
        if self.last_distance \
                and self.last_duration \
                and self.start_pos.any() \
                and self.last_jump_right is not None:
            pass
        else:
            return

        # 计算棋子和起始棋盘中心的偏差距离
        d = self.calc_distance(self.start_pos, self.piece_pos, self.last_jump_right)

        # 计算实际跳跃距离，这里要分情况讨论跳过头和没跳到两种情况讨论
        k = 1 / math.sqrt(3) if self.last_jump_right else -1 / math.sqrt(3)
        # 没跳到，实际距离 = 上次测量距离 - 偏差距离
        if self.piece_pos[1] > k*(self.piece_pos[0]-self.start_pos[0]) + self.start_pos[1]:
            self.last_actual_distance = self.last_distance - d
        # 跳过头，实际距离 = 上次测量距离 + 偏差距离
        elif self.piece_pos[1] < k*(self.piece_pos[0]-self.start_pos[0]) + self.start_pos[1]:
            self.last_actual_distance = self.last_distance + d
        # 刚刚好
        else:
            self.last_actual_distance = self.last_distance

        print(self.last_actual_distance, self.last_duration, self.on_center)

    def get_target_img(self, img):
        """获取当前目标棋盘的图像。"""
        half_height = self.target_pos[1] - self.top_pos[1]
        half_width = int(round(half_height * math.sqrt(3)))
        self.target_img = img[
            self.target_pos[1]: self.target_pos[1]+half_height+100,
            self.target_pos[0]-half_width-3: self.target_pos[0]+half_width+3,
        ]
        return self.target_img

    def jump(self):
        """跳跃，并存储本次目标跳跃距离和按压时间"""
        # 计算棋子坐标和目标棋盘中心坐标之间距离
        self.distance = self.calc_distance(self.piece_pos, self.target_pos, self.jump_right)
        # 计算长按时间
        self.duration = int(round(self.model.predict(self.distance)))
        # 跳！
        self.adb.long_tap(self.resolution // 2, self.duration)

    def mark_img(self, img_rgb):
        """在 RGB 图像上标注数据，会改变传入的 img_rgb"""
        draw = ImageDraw.Draw(img_rgb)
        # 棋子中心点，红色
        draw.line((0, self.piece_pos[1], self.resolution[0], self.piece_pos[1]), "#ff0000")
        draw.line((self.piece_pos[0], 0, self.piece_pos[0], self.resolution[1]), "#ff0000")
        # 目标棋盘中心点，蓝色
        draw.line((0, self.target_pos[1], self.resolution[0], self.target_pos[1]), "#0000ff")
        draw.line((self.target_pos[0], 0, self.target_pos[0], self.resolution[1]), "#0000ff")
        # 当前棋盘中心点，黑色
        draw.line((0, self.start_pos[1], self.resolution[0], self.start_pos[1]), "#000000")
        draw.line((self.start_pos[0], 0, self.start_pos[0], self.resolution[1]), "#000000")

        draw.multiline_text(
            (20, 20),
            "\n".join([
                "[上次跳跃数据]",
                f"向右跳跃: {self.last_jump_right}",            # noqa
                f"落点在中心: {self.on_center}",                 # noqa
                f"应跳跃距离: {self.last_distance}",             # noqa
                f"实跳跃距离: {self.last_actual_distance}",      # noqa
                f"按压时间: {self.last_duration}",              # noqa
            ]),
            fill='#000000',
            font=ImageFont.truetype(TTF_FONT_FILE, 45)
        )

    def single_run(self):
        """单次运行"""
        img_rgb = self.adb.screencap()
        img = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2GRAY)
        self.init_attrs()
        self.get_piece_pos(img)
        self.get_target_pos(img)
        self.get_start_pos(img)
        self.get_target_img(img)
        self.review_last_jump()
        self.jump()
        return img_rgb

    def run(self, jump_delay=1.1, show_img=False):
        """循环运行"""
        while True:
            img_rgb = self.single_run()
            time.sleep(self.duration/5000 + jump_delay)
            if show_img:
                self.mark_img(img_rgb)
                img_rgb.show()
