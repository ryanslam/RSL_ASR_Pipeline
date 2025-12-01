import argparse
import yaml
import time
import logging

from config.available_types import WHISPER_MODELS, ZMQ_PROTOCOLS
from lib.zmq_publisher import publish_to, close_socket
from ASR import ASRInitializer


def main():
    parser = init_arg_parser()
    args = parser.parse_args()
    conf = load_config(args)

    if conf["asr"].get("silence_logs", False):
        logging.getLogger("whisperx").setLevel(logging.ERROR)
        logging.getLogger("whisperx.asr").setLevel(logging.ERROR)
        logging.getLogger("whisperx.vads").setLevel(logging.ERROR)
        logging.getLogger("whisperx.vads.silero").setLevel(logging.ERROR)

    wm = conf["asr"]["whisper_model"]
    if wm not in WHISPER_MODELS:
        raise ValueError(
            f"Invalid whisper model '{wm}'. "
            f"Choose from: {', '.join(WHISPER_MODELS)}"
        )

    zmq_conf = conf["zmq"]
    speech_pub = None

    if zmq_conf["publish_text"]:
        if not zmq_conf["protocol"] in ZMQ_PROTOCOLS:
            raise ValueError(
                f"Invalid transport protocol '{zmq_conf['protocol']}'."
                f"Choose from: {', '.join(ZMQ_PROTOCOLS)}"
            )
        speech_pub = publish_to(
            protocol=zmq_conf["protocol"],
            addr=zmq_conf["addr"],
            port=zmq_conf["port"],
            bind=zmq_conf["bind"],
        )
        # Small delay to allow subscribers to connect (ZMQ slow joiner problem)
        if zmq_conf["bind"]:
            time.sleep(0.1)

    asr_conf = conf["asr"]
    ASR = ASRInitializer(
        whisper_model=asr_conf["whisper_model"],
        min_silence_sec=asr_conf["eou_silence"],
        lang_code=asr_conf["lang_code"],
        dev_idx=asr_conf["mic"],
    )

    stream = ASR.build_input_stream()
    stream.start()
    print("Listening... Press Ctrl+C to stop.\n")

    try:
        while True:
            if ASR.user_speech:
                text = ASR.user_speech[-1]["text"]
                print(f"Transcribed Speech: {text}")
                if speech_pub:
                    topic = zmq_conf.get("topic", "")
                    # Use send_multipart for proper PUB/SUB pattern if topic is set
                    # If no topic, send plain string (subscriber should subscribe to "")
                    if topic:
                        speech_pub.send_multipart([topic.encode(), text.encode()])
                    else:
                        speech_pub.send_string(text)
                    print(f"Publishing: {text}\n")

            ASR.user_speech = None
            time.sleep(0.1)

    except KeyboardInterrupt:
        stream.stop()
        if speech_pub:
            close_socket(speech_pub)
        print("ASR Stopped")


def init_arg_parser():
    parser = argparse.ArgumentParser(
        prog="Active Speech Recognizer",
        description="Transcribes user speech and optionally publishes text via ZMQ.",
    )

    parser.add_argument(
        "--config",
        type=str,
        metavar="",
        default="./config/config.yaml",
        help="YAML config file.",
    )

    # ZMQ overrides
    parser.add_argument(
        "--publish_text",
        type=bool,
        metavar="",
        help="Enables speech transcription publishing via ZMQ.",
    )
    parser.add_argument(
        "--zmq_protocol",
        type=str,
        metavar="",
        help=f"Set the protocol. Available options: {ZMQ_PROTOCOLS}",
    )
    parser.add_argument(
        "--zmq_addr", type=str, metavar="", help="IP address to publish to."
    )
    parser.add_argument(
        "--zmq_port", type=int, metavar="", help="Specified port to publish to."
    )
    parser.add_argument(
        "--zmq_bind",
        type=bool,
        metavar="",
        help="Select if you want the publisher to bind.",
    )
    parser.add_argument(
        "--zmq_topic", type=str, metavar="", help="Specific ZMQ topic to publish to."
    )

    # ASR overrides
    parser.add_argument(
        "--eou_silence",
        type=float,
        metavar="",
        help="Duration of silence required to begin transcribing.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        metavar="",
        help=f"Transcription model. Options are: {WHISPER_MODELS}",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        metavar="",
        help="Target sample rate. VAD requires 16khz.",
    )
    parser.add_argument(
        "--chunk_size", type=int, metavar="", help="Chunk size to be sent."
    )
    parser.add_argument(
        "--mic",
        type=int,
        metavar="",
        help="Microphone index for selected input device.",
    )
    parser.add_argument(
        "--silence_logs",
        type=bool,
        metavar="",
        help="Silences debugging information from whisper.",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        metavar="",
        help="Impose language bias upon whisper transcription model."
    )

    return parser


def load_config(args):
    if args.config:
        with open(args.config, "r") as f:
            conf = yaml.safe_load(f)
    else:
        conf = {
            "zmq": {
                "publish_text": True,
                "protocol": "tcp",
                "addr": "0.0.0.0",
                "port": 5555,
                "bind": True,
                "topic": "transcribed_speech",
            },
            "asr": {
                "eou_silence": 1.5,
                "whisper_model": "base",
                "target_sr": 16000,
                "chunk_size": 512,
                "mic": 10,
                "silence_logs": True,
                "lang_code": "en"
            },
        }

    overrides = {
        "zmq.publish_text": args.publish_text,
        "zmq.protocol": args.zmq_protocol,
        "zmq.addr": args.zmq_addr,
        "zmq.port": args.zmq_port,
        "zmq.bind": args.zmq_bind,
        "zmq.topic": args.zmq_topic,
        "asr.eou_silence": args.eou_silence,
        "asr.whisper_model": args.whisper_model,
        "asr.target_sr": args.target_sr,
        "asr.chunk_size": args.chunk_size,
        "asr.mic": args.mic,
        "asr.silence_logs": args.silence_logs,
        "asr.lang_code": args.lang_code,
    }

    for key, val in overrides.items():
        if val is None:
            continue
        section, field = key.split(".")
        conf[section][field] = val

    return conf


if __name__ == "__main__":
    main()
