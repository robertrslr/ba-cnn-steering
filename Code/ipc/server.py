
"""
Unix Domain Socket Server 


"""

import socket 
import sys 
import os

server_address= "/tmp/caroloIPC.uds"

#Falls der socket schon existiert, versuche ihn zu unlinken
try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise Exception
        
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
# Bind the socket to the port
print >>sys.stderr, 'starting up on %s' % server_address
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()
    try:
        print >>sys.stderr, 'connection from', client_address
        connection.sendall("Hello from Python Server")
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(64)
            print >>sys.stderr, 'received "%s"' %  str(data)
            if data:
                msg = input("Nachricht eingeben: ")
                print >>sys.stderr, 'Nachricht wird an client gesendet'
                connection.sendall(msg)
            else:
                print >>sys.stderr, 'no more data from', client_address
                break
            
    finally:
        # Clean up the connection
        connection.close()

