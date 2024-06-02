#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of reading and writing simulaneously from a serial port. The message that is sent to the
device is sent back, read in the receive thead and printed to console.

@author: G. Howell
"""

import serial
import threading
import time

ser = None
sendMsg = b'hello world\n'

def txThread():
    while ser:
        ser.write(sendMsg)
        time.sleep(3)

def rxThread():
    while ser:
        nRdBytes = ser.inWaiting()
        if nRdBytes > 0:
            receiverMsg = (ser.read(nRdBytes))
            print((receiverMsg.decode("utf-8"))[0:-1])
        else:
            time.sleep(0.1)

def loopbackTest(portName):
    global ser

    ser = serial.Serial(portName, 115200, timeout=1)

    threading.Thread(target=txThread).start()
    threading.Thread(target=rxThread).start()

    try:
        while True:
            time.sleep(1)
    except:
        ser = None

if __name__ == "__main__":
    
    loopbackTest(portName='/dev/tty.usbserial-0001')
