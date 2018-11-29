# -*- coding: utf-8 -*-
import socket
import os

steering_value = 0.0


print("Connecting...")
if os.path.exists("/tmp/caroloIPC.uds"):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect("/tmp/caroloIPC.uds")
    print("Ready.")
    print("Ctrl-C to quit.")
    print("Sending 'DONE' shuts down the server and quits.")
    while True:
        try:
            client.send(steering_value)
          
        except KeyboardInterrupt as k:
            print("Shutting down.")
            client.close()
            break
else:
    print("Couldn't Connect!")
print("Done")

