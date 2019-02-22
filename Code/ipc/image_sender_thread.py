# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:30:47 2019

@author: Jan Robert Rösler
"""

import socket 
import sys
import threading
import Queue

class ImageSenderThread(threading.Thread):
    """
    Sends the image capture from the ueye camera to a server.
    """
    queue = Queue.Queue()
    
    
    def __init__(self,queue,out_queue):
        threading.Thread.__init__(self)
        self.queue = queue
        
        
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = ""
        port = ""
        
        try:
            soc.connect((host, port))
        except:
            print("Connection error")
            sys.exit()
    
    def run(self):
        #get image from queue if not empty
        image_to_send = self.queue.get()
        #prepare image to be sent
        bytes = image_to_send.read()
        size = len(bytes)
        
        
        
        
        
        
