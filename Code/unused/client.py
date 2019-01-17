"""
Unix Domain Socket Client 

"""

import socket
import sys

# Create a UDS socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = "/tmp/caroloIPC.uds"
print >>sys.stderr, 'connecting to %s' % server_address

try:
    sock.connect(server_address)
except socket.error as msg:
    print(sys.stderr, msg)
    sys.exit(1)
    
try:
    while True: 
        data = sock.recv(64)
        print >>sys.stderr, 'received "%s"' % data
        if data: 
            msg = input ("Nachricht an Server eingeben: ")
            print >>sys.stderr, 'sending "%s"' % msg
            sock.sendall(msg)
        else:
            print >>sys.stderr, 'server gone, no more data'	
            break 
    
finally:
    print >>sys.stderr, 'closing socket'
    sock.close()
