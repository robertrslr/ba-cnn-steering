import time
import threading
import socket
import os

class uds_socket_thread(threading.Thread):

    def __init__(self, socket_path="/tmp/caroloIPC.uds",socket_family=socket.AF_UNIX,
                 socket_type=socket.SOCK_STREAM):

        if os.path.exists(socket_path):
            self.client = socket.socket(socket_family, socket_type)
            self.client.connect(socket_path)
            print("Ready.")
            print("Ctrl-C to quit.")
        else:
            print("Couldn't Connect!")
        print("Done")

    def run(self):
        while True:
            try:
                self.client.send(steering_value)

            except KeyboardInterrupt as k:
                print("Shutting down.")
                self.client.close()
                break

