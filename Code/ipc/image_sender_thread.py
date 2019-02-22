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
    data = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    
    def __init__(self,queue,sock):
        threading.Thread.__init__(self)
        self.queue = queue
        self.sock = sock
        
        
        
        host = ""
        port = ""
        
        try:
            sock.connect((host, port))
        except:
            print("Connection error")
            sys.exit()
    
    def run(self):
        
        while True:
            #get image from queue if not empty
            image_to_send = self.queue.get()
            #prepare image to be sent
            bytes = image_to_send.read()
            size = len(bytes)
            
            self.sock.sendall("SIZE %" % size)
            answer = self.sock.recv(4096)
            # send image to server
            if answer == 'GOT SIZE':
                sock.sendall(bytes)

                # check what server send
                answer = sock.recv(4096)
                print(answer = %s' % answer)
                
                if answer == 'GOT IMAGE' :

        
        
    def __del__(self):
        self.sock.close()
