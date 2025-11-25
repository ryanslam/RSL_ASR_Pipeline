import zmq

context = zmq.Context()

subscriber = context.socket(zmq.SUB)

subscriber.connect("tcp://0.0.0.0:5555")

subscriber.setsockopt_string(zmq.SUBSCRIBE, "transcribed_speech")

print(
    "Subscriber connected and waiting for messages...\n\
    Press Ctrl+C to stop listening."
)

try:
    while True:
        parts = subscriber.recv_multipart()
        if len(parts) == 2:
            topic = parts[0].decode()
            message = parts[1].decode()
            print(f"Received [{topic}]: {message}")
        elif len(parts) == 1:
            message = parts[0].decode()
            print(f"Received: {message}")

except KeyboardInterrupt:
    print("Subscriber stopped.")
finally:
    subscriber.close()
    context.term()
