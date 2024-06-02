#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write parameters to device.

@author: G. Howell
"""

import serial, threading, time

class paramWriter():
    
    def __init__(self,
                 portName):
        
        # create dummy set of parameters
        self.params = {
            'val1' : 14
        }
        
        # initialise serial port
        self.ser = serial.Serial(portName, 115200, timeout=1)

    def txThread(self):
        
        while self.ser:
            # send all the parameters one by one
            for param in self.params:
                
                # create and send the parameter data, appending with the newline character to signal end of message on device
                sendMsg = bytes(f'{param} : {self.params[param]}\n', 'utf-8')
                self.ser.write(sendMsg)
                print(f'Bytes sent: {len(sendMsg)}')
                
                # pause between sending params
                time.sleep(5)

            # null the serial device to signal end of transmission
            self.ser = None

    def rxThread(self):
        
        # hold thead open until the tx thead has finished sending messages
        while self.ser:
            # read return message to see if data was received correctly
            nRdBytes = self.ser.inWaiting()
            if nRdBytes > 0:
                receiverMsg = (self.ser.read(nRdBytes))
                print(f'Bytes received: {nRdBytes}')
                print((receiverMsg.decode("utf-8"))[0:-1])
            else:
                time.sleep(0.1)
            
    def writeParams(self):
        
        # open theads
        threading.Thread(target=self.txThread).start()
        threading.Thread(target=self.rxThread).start()
        
        try:
            while self.ser != None:
                time.sleep(1)
        except:
            self.ser = None

if __name__ == "__main__":
    
    portName = '/dev/tty.usbserial-0001'
    
    pw = paramWriter(portName)
    
    pw.writeParams()
    