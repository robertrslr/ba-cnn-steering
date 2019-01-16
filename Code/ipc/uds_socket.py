# -*- coding: utf-8 -*-
import socket
import os
import struct
from datetime import date


class uds_socket():
    def __init__(self,socket_path="/tmp/caroloIPC.uds", socket_family=socket.AF_UNIX,
                 socket_type=socket.SOCK_STREAM):
        if os.path.exists(socket_path):
            self.client = socket.socket(socket_family, socket_type)
            self.client.connect(socket_path)
            print("UDS Socket Client connected to", socket_path)
            print("Ctrl-C to quit.")
        else:
            print("Couldn't Connect!")

    def send_data(self,data,pack = True):

        try:
            if pack:
                packed_data = str(data)
                self.client.send(packed_data.encode('utf-8'))
            else:
                self.client.send(data)
                #print(data)
        except KeyboardInterrupt as k:
            print("Shutting down.")
            self.client.close()

    def pack_float_value(self, float):
        """
        Packs a float value in a struct.
        :param data:
        :return:
        """
        packed_float = struct.pack("<f", float)
        return packed_float

    def __del__(self):
        self.client.send("quit\n".encode('utf-8'))
        self.client.close()