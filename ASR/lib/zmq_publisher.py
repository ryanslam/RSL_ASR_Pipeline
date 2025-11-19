import zmq

def publish_to(addr='0.0.0.0', port=None, bind=False) -> zmq.Socket:
    context = zmq.context()
    socket = context.socket(zmq.PUB)

    if port:
        addr = f"{addr}:{port}"
    if bind:
        socket.bind(addr)
    else:
        socket.connect(addr)

    return socket

def close_socket(socket) -> None:
    socket.close()
    socket.context.term()
    
    return None