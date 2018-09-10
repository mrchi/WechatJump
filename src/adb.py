#!/usr/bin/env python3
# coding=utf-8

import functools
import subprocess
from io import BytesIO

from PIL import Image

__all__ = ["PyADB"]

_sysrun = functools.partial(
    subprocess.run,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)


class ADBError(Exception): pass


class ConnectionError(ADBError): pass


class LongTapError(ADBError): pass


class ShortTapError(ADBError): pass


class PyADB:

    def __init__(self, device_serial):
        self.device_serial = device_serial

    def connect(self, ip, port=5555):
        """连接网络adb调试设备"""
        cmd = f"adb -s {self.device_serial} connect {ip}:{port}".split()
        try:
            result = _sysrun(cmd, timeout=2)
            returncode = result.returncode
        except subprocess.TimeoutExpired as e:
            errmsg = f"Connect {ip}:{port} timeout."
        else:
            if result.returncode != 0:
                raise ConnectionError(result.stderr.decode.strip())
            return "connected" in output, output
        raise ConnectionError(errmsg)

    def get_resolution(self):
        """获取屏幕分辨率"""
        cmd = f"adb -s {self.device_serial} exec-out wm size".split()
        result = _sysrun(cmd)
        w, h = result.stdout.decode().split("Physical size: ")[-1].split("x")
        return (int(w), int(h))

    def screencap(self):
        """截图，输出为 Pillow.Image 对象"""
        cmd = f"adb -s {self.device_serial} exec-out screencap -p".split()
        result = _sysrun(cmd)
        img = Image.open(BytesIO(result.stdout))
        return img

    def short_tap(self, cord):
        """短按点击，坐标为 (x, y) 格式"""
        cmd = f"adb -s {self.device_serial} exec-out input tap {cord[0]} {cord[1]}".split()
        result = _sysrun(cmd)
        if result.returncode != 0:
            raise ShortTapError(result.stderr.decode.strip())

    def long_tap(self, cord, duration):
        """长按, duration单位为ms，坐标为 (x, y) 格式"""
        cmd = f"adb -s {self.device_serial} exec-out input swipe {cord[0]} {cord[1]} {cord[0]} {cord[1]} {duration}".split()
        result = _sysrun(cmd)
        if result.returncode != 0:
            raise LongTapError(result.stderr.decode.strip())
