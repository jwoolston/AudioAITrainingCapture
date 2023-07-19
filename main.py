# This is a sample Python script.
import signal
import sys

from AudioHandler import AudioHandler
from Gui import Gui


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    audio.stop()
    gui.stop()
    sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    audio = AudioHandler()
    gui = Gui(audio)

    signal.signal(signal.SIGINT, signal_handler)

    audio.start()  # open the stream
    gui.run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
