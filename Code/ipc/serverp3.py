#-*- coding: utf-8 -*-
import socket
import os

if os.path.exists("/tmp/python_unix_sockets_example"):
    os.remove("/tmp/python_unix_sockets_example")

print("Opening socket...")
server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
server.bind("/tmp/python_unix_sockets_example")

print("Listening...")
x = "Hello from Python Server"
i = 1
while True:
    datagram = server.recv(1024)
    if not datagram:
        break
    else:
        print("-" * 20)
        print(datagram.decode('utf-8'))
        if i>0:
            server.send(x.encode('utf-8'))
            i=0
        msg = input("> ")
        if "" != x:
            print("SEND:", x)
            server.send(x.encode('utf-8'))
        if "DONE" == datagram.decode('utf-8'):
            break
print("-" * 20)
print("Shutting down...")
server.close()
os.remove("/tmp/python_unix_sockets_example")
print("Done")
