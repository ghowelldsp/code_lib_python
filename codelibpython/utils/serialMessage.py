#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enables communication with serial devices using Virtual Comp Ports

@author: G. Howell
"""

import serial

s = serial.Serial('/dev/tty.usbserial-0001', 115200, )
res = s.write(b'hii\r')
print(res)
