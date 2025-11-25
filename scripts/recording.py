import pyaudio
import wave
import argparse
import time
import os

class Listener:
    def __init__(self, args):
        self.chunk = 1024
        self.FORMAT = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = args.sample_rate 
        self.record_seconds = args.seconds  

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def save_audio(self, file_name, frames):
        print(f'Saving file to {file_name}')
        self.stream.stop_stream()
        self.stream.close()

        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        
        with wave.open(file_name, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))
        
        self.p.terminate()

def interactive(args):
    try:
        while True:
            listener = Listener(args)
            frames = []
            input(f'Press Enter to record {args.seconds} second(s). Ctrl+C to exit.')
            time.sleep(0.2)
            for i in range(int((listener.sample_rate / listener.chunk) * listener.record_seconds)):
                data = listener.stream.read(listener.chunk, exception_on_overflow=False)
                frames.append(data)
            save_path = os.path.join(args.interactive_save_path, f"{args.name}_{args.index}.wav")
            listener.save_audio(save_path, frames)
            args.index += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))

def main(args):
    listener = Listener(args)
    frames = []
    print('Recording...')
    try:
        if args.seconds is None:
            while True:
                print('Recording indefinitely... Ctrl+C to stop.', end="\r")
                data = listener.stream.read(listener.chunk, exception_on_overflow=False)
                frames.append(data)
        else:
            for i in range(int((listener.sample_rate / listener.chunk) * listener.record_seconds)):
                data = listener.stream.read(listener.chunk, exception_on_overflow=False)
                frames.append(data)
            raise Exception('Done recording.')
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))

    print('Finished recording.')
    listener.save_audio(args.save_path, frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script to collect data for wake word training.

    By default, it records 1 second at 16kHz.
    ''')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate for recording. Default=16000 Hz.')
    parser.add_argument('--seconds', type=int, default=1,
                        help='Recording duration in seconds. Default=1 second.')
    parser.add_argument('--save_path', type=str, default=None, required=False,
                        help='Full path to save single recording (e.g. ./audio.wav).')
    parser.add_argument('--interactive_save_path', type=str, default=None, required=False,
                        help='Directory to save multiple samples in interactive mode.')
    parser.add_argument('--interactive', default=False, action='store_true', required=False,
                        help='Enable interactive mode (record many samples).')
    parser.add_argument('--name', type=str, default=None, required=True,
                        help='name of file')
    parser.add_argument('--index', type=int, default=0, required=False,
                        help='start index')
    args = parser.parse_args()

    if args.interactive:
        if args.interactive_save_path is None:
            raise Exception('You must set --interactive_save_path when using --interactive.')
        interactive(args)
    else:
        if args.save_path is None:
            raise Exception('You must set --save_path.')
        main(args)
