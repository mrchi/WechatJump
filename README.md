# WechatJump

## 环境搭建

- Python 3.5+
- 安装依赖

 ```shell
 brew install opencv
 pipenv install
 ```

## 原理

1. 使用 adb 截图；
2. opencv 模版匹配确定棋子坐标；
3. opencv 模版匹配确定棋盘中心小白点，如果没有小白点，使用边缘检测确定棋盘位置；
4. 根据棋子和目标中心的距离计算按压时间；
5. 使用 adb 操作按压；

## 成绩

目前最好成绩在 800+。

原因：公式存在偏差，目前使用 `time = k * distance` 而不是 `time = k * distance + b`；

## TODO

[] 对棋子实际落地点进行采集；

[] 使用线性回归优化公式；